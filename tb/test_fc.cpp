#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>

#include "ap_int.h"
#include "hls4pc.hpp"

using namespace std;

#define PE_COUNT       4
#define SIMD_WIDTH     8
#define INPUT_SIZE     128
#define OUTPUT_SIZE    64

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
#define WEIGHT_MEM_SIZE ((INPUT_SIZE * OUTPUT_SIZE) / (SIMD_WIDTH * PE_COUNT))
#define BIAS_MEM_COLS   (OUTPUT_SIZE / PE_COUNT)

template<int W>
vector< ap_uint<W> > read_hex_file(const string& fn) {
    vector< ap_uint<W> > v;
    ifstream f(fn);
    if (!f) {
        cerr << "Failed to open " << fn << "\n";
        return v;
    }
    string line;
    while (getline(f, line)) {
        if (line.empty()) continue;
        ap_uint<W> val = 0;
        for (char c : line) {
            unsigned d = 0;
            if      (c >= '0' && c <= '9') d = c - '0';
            else if (c >= 'a' && c <= 'f') d = 10 + c - 'a';
            else if (c >= 'A' && c <= 'F') d = 10 + c - 'A';
            else continue;
            val = (val << 4) | d;
        }
        v.push_back(val);
    }
    return v;
}

int main() {
    constexpr int IN_BITS     = SIMD_WIDTH * INPUT_PRECISION;
    constexpr int IN_HEX_DIG  = (IN_BITS  + 3) / 4;
    constexpr int OUT_HEX_DIG = (OUTPUT_PRECISION + 3) / 4;
    const uint64_t BITMASK = (OUTPUT_PRECISION < 64
                              ? ((1ULL << OUTPUT_PRECISION) - 1)
                              : 0xFFFFFFFFFFFFFFFFULL);

    hls::stream< ap_uint<IN_BITS> > in_strm;
    hls::stream< ap_uint<PE_COUNT * OUTPUT_PRECISION> > out_strm;

    auto in_data = read_hex_file<IN_BITS>("q_81_input.txt");
    if (in_data.size() < (INPUT_SIZE / SIMD_WIDTH)) {
        cerr << "Not enough input words\n";
        return 1;
    }
    for (auto &w : in_data) in_strm.write(w);

    auto w_data = read_hex_file<WEIGHTS_PRECISION>("q_81_weights.txt");
    if (w_data.size() < PE_COUNT * SIMD_WIDTH * WEIGHT_MEM_SIZE) {
        cerr << "Not enough weight words\n";
        return 1;
    }
    ap_uint<WEIGHTS_PRECISION> weightMem[PE_COUNT][SIMD_WIDTH][WEIGHT_MEM_SIZE];
    int idx = 0;
    for (int pe = 0; pe < PE_COUNT; ++pe)
      for (int sd = 0; sd < SIMD_WIDTH; ++sd)
        for (int i = 0; i < WEIGHT_MEM_SIZE; ++i)
          weightMem[pe][sd][i] = w_data[idx++];

    auto b_data = read_hex_file<BIAS_PRECISION>("q_81_biases.txt");
    if (b_data.size() < PE_COUNT * BIAS_MEM_COLS) {
        cerr << "Not enough bias words\n";
        return 1;
    }
    ap_uint<BIAS_PRECISION> biasMem[PE_COUNT][BIAS_MEM_COLS];
    idx = 0;
    for (int pe = 0; pe < PE_COUNT; ++pe)
      for (int i = 0; i < BIAS_MEM_COLS; ++i)
        biasMem[pe][i] = b_data[idx++];

    FCMac<INPUT_SIZE, OUTPUT_SIZE,
          PE_COUNT, SIMD_WIDTH,
          BIAS_PRECISION,  BIAS_INT_PRECISION,
          WEIGHTS_PRECISION, WEIGHTS_INT_PRECISION,
          INPUT_PRECISION, INPUT_INT_PRECISION,
          MUL_PRECISION,   MUL_INT_PRECISION,
          ACC_PRECISION,   ACC_INT_PRECISION,
          OUTPUT_PRECISION, OUTPUT_INT_PRECISION
    >(in_strm, out_strm, weightMem, biasMem);

    auto gt = read_hex_file<OUTPUT_PRECISION>("q_81_output.txt");
    vector< ap_uint<OUTPUT_PRECISION> > hw;
    while (!out_strm.empty()) {
        auto wide = out_strm.read();
        for (int pe = 0; pe < PE_COUNT; ++pe) {
            int lo = pe * OUTPUT_PRECISION;
            hw.push_back(wide.range(lo + OUTPUT_PRECISION - 1, lo));
        }
    }

    size_t n = min(gt.size(), hw.size());
    cout << "Idx\tGT Hex\tGT Float\tHW Hex\tHW Float\n";
    int frac = PRECISION - INT_PRECISION;
    double mse = 0;
    for (size_t i = 0; i < n; ++i) {
        ap_int<OUTPUT_PRECISION> g = (ap_int<OUTPUT_PRECISION>)gt[i];
        ap_int<OUTPUT_PRECISION> h = (ap_int<OUTPUT_PRECISION>)hw[i];
        double gf = double((int64_t)g) / (1 << frac);
        double hf = double((int64_t)h) / (1 << frac);
        mse += (gf - hf)*(gf - hf);
        uint64_t gb = uint64_t(gt[i]) & BITMASK;
        uint64_t hb = uint64_t(hw[i]) & BITMASK;
        cout << dec << setw(3) << i << "\t"
             << "0x" << hex << setw(OUT_HEX_DIG) << setfill('0') << gb << dec << "\t"
             << fixed << setprecision(5) << gf << "\t"
             << "0x" << hex << setw(OUT_HEX_DIG) << setfill('0') << hb << dec << "\t"
             << fixed << setprecision(5) << hf << "\n";
    }
    cout << "\nMSE = " << (mse / n) << "\n";
    return 0;
}
