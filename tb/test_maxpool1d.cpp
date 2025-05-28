#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>

#include "ap_int.h"
#include "maxpool1d.cpp"  

using namespace std;

#define KERNEL_DIM    2
#define IFM_CHANNELS  64
#define IFM_DIM       4096
#define STRIDE        2
#define PADDING       0
#define OFM_CHANNELS  IFM_CHANNELS
#define OFM_DIM       (IFM_DIM / KERNEL_DIM)
#define SIMD_WIDTH    8
#define PRECISION      32
#define INT_PRECISION   16
#define INPUT_PRECISION       PRECISION
#define INPUT_INT_PRECISION   INT_PRECISION
#define OUTPUT_PRECISION      PRECISION
#define OUTPUT_INT_PRECISION  INT_PRECISION
#define SYNAPSE_FOLD  (IFM_CHANNELS / SIMD_WIDTH)

template<int W>
vector< ap_uint<W> > read_hex_file(const string& fn) {
    vector< ap_uint<W> > v;
    ifstream f(fn);
    if (!f) { cerr<<"Failed to open "<<fn<<"\n"; return v; }
    string line;
    while (getline(f, line)) {
        if (line.empty()) continue;
        ap_uint<W> val = 0;
        for (char c : line) {
            unsigned d = 0;
            if      (c>='0'&&c<='9') d = c - '0';
            else if (c>='a'&&c<='f') d = 10 + c - 'a';
            else if (c>='A'&&c<='F') d = 10 + c - 'A';
            else continue;
            val = (val << 4) | d;
        }
        v.push_back(val);
    }
    return v;
}

int main() {
    constexpr int IN_BITS  = SIMD_WIDTH * INPUT_PRECISION;
    constexpr int OUT_BITS = SIMD_WIDTH * OUTPUT_PRECISION;
    const uint64_t BITMASK = (OUTPUT_PRECISION<64
                              ? ((1ULL<<OUTPUT_PRECISION)-1)
                              : 0xFFFFFFFFFFFFFFFFULL);

    hls::stream< ap_uint<IN_BITS> >  in_strm;
    hls::stream< ap_uint<OUT_BITS> > out_strm;

    auto in_data = read_hex_file<IN_BITS>("q_3216_input.txt");
    if (in_data.size() < (size_t)(IFM_DIM * SYNAPSE_FOLD)) {
        cerr<<"Not enough input words\n"; return 1;
    }
    for (auto &w : in_data) in_strm.write(w);

    MAXPool1D<
      KERNEL_DIM, IFM_CHANNELS, IFM_DIM,
      STRIDE, PADDING,
      OFM_CHANNELS, OFM_DIM,
      SIMD_WIDTH,
      INPUT_PRECISION, INPUT_INT_PRECISION,
      OUTPUT_PRECISION, OUTPUT_INT_PRECISION
    >(in_strm, out_strm);

    vector< ap_uint<OUTPUT_PRECISION> > hls_lanes;
    while (!out_strm.empty()) {
        auto wide = out_strm.read();
        for (int lane = 0; lane < SIMD_WIDTH; ++lane) {
            int lo = lane * OUTPUT_PRECISION;
            ap_uint<OUTPUT_PRECISION> v = wide.range(lo + OUTPUT_PRECISION - 1, lo);
            hls_lanes.push_back(v);
        }
    }

    auto gt = read_hex_file<OUTPUT_PRECISION>("q_3216_output.txt");
    size_t n = min(gt.size(), hls_lanes.size());

    cout<<"Idx\tGT Hex\tGT Float\tHW Hex\tHW Float\n";
    int frac = PRECISION - INT_PRECISION;
    double mse = 0;
    for (size_t i = 0; i < n; ++i) {
        int64_t g_signed = (ap_int<OUTPUT_PRECISION>)gt[i];
        int64_t h_signed = (ap_int<OUTPUT_PRECISION>)hls_lanes[i];
        double gf = double(g_signed) / (1<<frac);
        double hf = double(h_signed) / (1<<frac);
        mse += (gf - hf)*(gf - hf);

        uint64_t gb = uint64_t(gt[i]) & BITMASK;
        uint64_t hb = uint64_t(hls_lanes[i]) & BITMASK;

        cout<<setw(5)<<setfill('0')<<i<<"\t"
            <<"0x"<<hex<<setw((PRECISION+3)/4)<<setfill('0')<<gb<<dec<<"\t"
            <<fixed<<setprecision(5)<<gf<<"\t"
            <<"0x"<<hex<<setw((PRECISION+3)/4)<<setfill('0')<<hb<<dec<<"\t"
            <<fixed<<setprecision(5)<<hf<<"\n";
    }
    cout<<"\nMSE = "<<(mse/n)<<"\n";
    return 0;
}
