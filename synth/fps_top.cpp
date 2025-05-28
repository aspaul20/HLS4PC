#ifndef __SYNTHESIS__
#include <hls_stream.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <iostream>
#include <fstream>
#include <vector>
#endif

#include "hls4pc.hpp"

#define PRECISION 8
#define INT_PREC 4
#define N 512
#define NPOINT 256

extern "C" void fps_top(
    hls::stream<Point<PRECISION, INT_PREC> > &in_strm,
    hls::stream<ap_uint<16> > &out_strm,
    int ground_truth
) {
    #pragma HLS INTERFACE axis port=in_strm
    #pragma HLS INTERFACE axis port=out_strm
    #pragma HLS INTERFACE s_axilite port=ground_truth
    #pragma HLS INTERFACE s_axilite port=return


    Point<PRECISION, INT_PREC> buffer[N];
    #pragma HLS ARRAY_PARTITION variable=buffer complete

    for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE
        buffer[i] = in_strm.read();
    }

    fps<PRECISION, INT_PREC, N, NPOINT, 4>(in_strm, out_strm, ground_truth);
}
