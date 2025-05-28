#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>

#include "ap_int.h"
#include "relu1d.cpp"   

using namespace std;

#define IFM_CHANNELS    64
#define IFM_DIM         4096
#define SIMD_WIDTH      8

#define PREC_I          32
#define INT_I           16
#define PREC_O          32
#define INT_O           16

#define SYNAPSE_FOLD    (IFM_CHANNELS / SIMD_WIDTH)

template<int W>
vector< ap_uint<W> > read_hex(const string& fn) {
    vector< ap_uint<W> > v;
    ifstream f(fn);
    if (!f) {
        cerr<<"Failed to open "<<fn<<"\n";
        return v;
    }
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
    constexpr int IN_BITS  = SIMD_WIDTH * PREC_I;
    constexpr int OUT_BITS = SIMD_WIDTH * PREC_O;
    const uint64_t MASK_O  = (PREC_O<64 ? ((1ULL<<PREC_O)-1) : 0xFFFFFFFFFFFFFFFFULL);

    hls::stream< ap_uint<IN_BITS> >  in_strm;
    hls::stream< ap_uint<OUT_BITS> > out_strm;

    auto in_data = read_hex<IN_BITS>("q_3216_input.txt");
    if (in_data.size() < (size_t)(IFM_DIM * SYNAPSE_FOLD)) {
        cerr<<"Not enough input words\n"; return 1;
    }
    for (auto &w : in_data) in_strm.write(w);

    Relu1D<
      IFM_CHANNELS, IFM_DIM, SIMD_WIDTH,
      PREC_I, INT_I,
      PREC_O, INT_O
    >(in_strm, out_strm);

    vector< ap_uint<PREC_O> > hw_lanes;
    while (!out_strm.empty()) {
        auto wide = out_strm.read();
        for (int lane = 0; lane < SIMD_WIDTH; ++lane) {
            int lo = lane * PREC_O;
            hw_lanes.push_back(wide.range(lo + PREC_O - 1, lo));
        }
    }

    auto gt_lanes = read_hex<PREC_O>("q_3216_output.txt");
    size_t n = min(gt_lanes.size(), hw_lanes.size());

    cout<<"Idx\tGT Hex\tGT Float\tHW Hex\tHW Float\n";
    int frac = PREC_O - INT_O;
    double mse = 0;
    for (size_t i = 0; i < n; ++i) {
        int64_t g_s = (ap_int<PREC_O>)gt_lanes[i];
        int64_t h_s = (ap_int<PREC_O>)hw_lanes[i];
        double gf = double(g_s) / (1<<frac);
        double hf = double(h_s) / (1<<frac);
        mse += (gf - hf)*(gf - hf);
        uint64_t gb = uint64_t(gt_lanes[i]) & MASK_O;
        uint64_t hb = uint64_t(hw_lanes[i]) & MASK_O;
        cout<<setw(5)<<setfill('0')<<i<<"\t"
            <<"0x"<<hex<<setw((PREC_O+3)/4)<<setfill('0')<<gb<<dec<<"\t"
            <<fixed<<setprecision(5)<<gf<<"\t"
            <<"0x"<<hex<<setw((PREC_O+3)/4)<<setfill('0')<<hb<<dec<<"\t"
            <<fixed<<setprecision(5)<<hf<<"\n";
    }
    cout<<"\nMSE = "<<(mse/n)<<"\n";
    return 0;
}
