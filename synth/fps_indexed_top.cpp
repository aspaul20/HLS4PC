#ifndef __SYNTHESIS__
#include <hls_stream.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <iostream>
#include <fstream>
#include <vector>
#endif

#include "hls4pc.hpp"
#define PRECISION 32
#define INT_PREC 16
#define N 512
#define NPOINT 256
#define UF 4

extern "C" void fps_indexed_top(
    hls::stream<Point<PRECISION, INT_PREC>> &in_strm,
    hls::stream<Point<PRECISION, INT_PREC>> &out_strm,
    ap_uint<16> init_farthest
) {
    #pragma HLS INTERFACE axis      port=in_strm
    #pragma HLS INTERFACE axis      port=out_strm
    #pragma HLS INTERFACE s_axilite port=init_farthest bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=return        bundle=CTRL

    fps_indexed<PRECISION, INT_PREC, N, NPOINT, UF>(
        in_strm,
        out_strm,
        init_farthest
    );
}
