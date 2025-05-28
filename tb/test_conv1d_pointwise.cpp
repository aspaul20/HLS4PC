#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>

#include "ap_int.h"
#include "hls_stream.h"
#include "hls4pc.hpp"

using namespace std;

#define KERNEL_DIM      1
#define IFM_CHANNELS    64
#define IFM_DIM         4096
#define STRIDE          1
#define PADDING         0
#define OFM_CHANNELS    64
#define OFM_DIM         4096
#define PE_COUNT        4
#define SIMD_WIDTH      1

#define PRECISION       8
#define INT_PRECISION    1

#define BIAS_PRECISION        PRECISION
#define BIAS_INT_PRECISION    INT_PRECISION
#define WEIGHTS_PRECISION     PRECISION
#define WEIGHTS_INT_PRECISION INT_PRECISION
#define INPUT_PRECISION       PRECISION
#define INPUT_INT_PRECISION   INT_PRECISION

#define MUL_PRECISION         (PRECISION*2)
#define MUL_INT_PRECISION     (INT_PRECISION*2)
#define ACC_PRECISION         (MUL_PRECISION + 6)
#define ACC_INT_PRECISION     (MUL_INT_PRECISION + 6)

#define OUTPUT_PRECISION      PRECISION
#define OUTPUT_INT_PRECISION  INT_PRECISION

#define WEIGHT_MEM_SIZE  ((KERNEL_DIM * IFM_CHANNELS * OFM_CHANNELS) / (SIMD_WIDTH * PE_COUNT))
#define BIAS_MEM_COLS    (OFM_CHANNELS / PE_COUNT)

template<int W>
using data_u = ap_uint<W>;

template<int W>
vector<data_u<W>> read_hex_file(const string& filename) {
   vector<data_u<W>> data;
   ifstream file(filename);
   if (!file) {
       cerr << "Failed to open " << filename << "\n";
       return data;
   }
   string line;
   const uint64_t mask = (W < 64 ? ((1ULL<<W)-1) : 0xFFFFFFFFFFFFFFFFULL);
   while (getline(file, line)) {
       if (line.empty()) continue;
       unsigned long long val = 0;
       stringstream ss(line);
       ss >> hex >> val;
       data.push_back( data_u<W>( val & mask ) );
   }
   return data;
}

int main() {
   constexpr int HEX_DIGITS = (PRECISION + 3) / 4;
   const uint64_t BITMASK = (PRECISION < 64 ? ((1ULL<<PRECISION)-1) : 0xFFFFFFFFFFFFFFFFULL);

   hls::stream< ap_uint<SIMD_WIDTH * INPUT_PRECISION> >  in_stream;
   hls::stream< ap_uint<SIMD_WIDTH * INPUT_PRECISION> >  buf_stream;
   hls::stream< ap_uint<PE_COUNT  * OUTPUT_PRECISION> > out_stream;

   auto input_data = read_hex_file<INPUT_PRECISION>("q_81_input.txt");
   for (auto &u : input_data) {
       ap_uint<SIMD_WIDTH*INPUT_PRECISION> tmp = u;
       in_stream.write(tmp);
   }

   auto weights_data = read_hex_file<WEIGHTS_PRECISION>("q_81_weights.txt");
   if (weights_data.size() < PE_COUNT*SIMD_WIDTH*WEIGHT_MEM_SIZE) {
       cerr << "Insufficient weight data\n";
       return 1;
   }
   ap_uint<WEIGHTS_PRECISION> weightMem[PE_COUNT][SIMD_WIDTH][WEIGHT_MEM_SIZE];
   int idx = 0;
   for (int pe=0; pe<PE_COUNT; ++pe)
     for (int s=0; s<SIMD_WIDTH; ++s)
       for (int i=0; i<WEIGHT_MEM_SIZE; ++i)
         weightMem[pe][s][i] = weights_data[idx++];

   auto biases_data = read_hex_file<BIAS_PRECISION>("q_81_biases.txt");
   if (biases_data.size() < PE_COUNT*BIAS_MEM_COLS) {
       cerr << "Insufficient bias data\n";
       return 1;
   }
   ap_uint<BIAS_PRECISION> biasMem[PE_COUNT][BIAS_MEM_COLS];
   idx = 0;
   for (int pe=0; pe<PE_COUNT; ++pe)
     for (int i=0; i<BIAS_MEM_COLS; ++i)
       biasMem[pe][i] = biases_data[idx++];

   Conv1DBuffer_new<
     KERNEL_DIM, IFM_CHANNELS, IFM_DIM,
     STRIDE,
     OFM_CHANNELS, OFM_DIM,
     PE_COUNT, SIMD_WIDTH,
     INPUT_PRECISION, INPUT_INT_PRECISION
   >(in_stream, buf_stream);

   Conv1DMac_new<
     KERNEL_DIM, IFM_CHANNELS, IFM_DIM,
     STRIDE, PADDING,
     OFM_CHANNELS, OFM_DIM,
     PE_COUNT, SIMD_WIDTH,
     BIAS_PRECISION,  BIAS_INT_PRECISION,
     WEIGHTS_PRECISION, WEIGHTS_INT_PRECISION,
     INPUT_PRECISION, INPUT_INT_PRECISION,
     MUL_PRECISION,   MUL_INT_PRECISION,
     ACC_PRECISION,   ACC_INT_PRECISION,
     OUTPUT_PRECISION, OUTPUT_INT_PRECISION
   >(buf_stream, out_stream, weightMem, biasMem);

   auto gt_output  = read_hex_file<OUTPUT_PRECISION>("q_81_output.txt");
   vector< ap_uint<OUTPUT_PRECISION> > hls_output;
   while (!out_stream.empty()) {
       auto wide = out_stream.read();
       for (int pe=0; pe<PE_COUNT; ++pe) {
           int lo = pe*OUTPUT_PRECISION;
           ap_uint<OUTPUT_PRECISION> v = wide.range(lo+OUTPUT_PRECISION-1, lo);
           hls_output.push_back(v);
       }
   }

   size_t n = min(gt_output.size(), hls_output.size());
   cout << "Index\tGT Hex\tGT Float\tHLS Hex\tHLS Float\n";

   int frac_bits = PRECISION - INT_PRECISION;
   double mse = 0.0;

   for (size_t i=0; i<n; ++i) {
       ap_int<OUTPUT_PRECISION> gt_s  = (ap_int<OUTPUT_PRECISION>)gt_output[i];
       ap_int<OUTPUT_PRECISION> hls_s = (ap_int<OUTPUT_PRECISION>)hls_output[i];

       double gt_f  = double((int64_t)gt_s)  / (1<<frac_bits);
       double hls_f = double((int64_t)hls_s) / (1<<frac_bits);

       mse += (gt_f - hls_f)*(gt_f - hls_f);


       uint64_t gt_bits  = uint64_t(gt_output[i])  & BITMASK;
       uint64_t hls_bits = uint64_t(hls_output[i]) & BITMASK;

       cout << dec << setw(5) << setfill('0') << i << "\t"
            << "0x" << hex << setw(HEX_DIGITS) << setfill('0') << gt_bits << dec << "\t"
            << fixed << setprecision(5) << gt_f << "\t"
            << "0x" << hex << setw(HEX_DIGITS) << setfill('0') << hls_bits << dec << "\t"
            << fixed << setprecision(5) << hls_f << "\n";
   }

   mse /= n;
   cout << "\nMSE = " << mse << "\n";
   return 0;
}
