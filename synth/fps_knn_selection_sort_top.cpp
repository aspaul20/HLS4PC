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
#define N        512
#define NPOINT   256
#define K        16
#define PEs      4

extern "C" void fps_knn_selection_sort_top(
    hls::stream<Point<PRECISION, INT_PREC>> &in_strm,
    hls::stream<Point<PRECISION, INT_PREC>> &grouped_strm,
    ap_uint<16> init_farthest
) {
    #pragma HLS INTERFACE axis      port=in_strm
    #pragma HLS INTERFACE axis      port=grouped_strm
    #pragma HLS INTERFACE s_axilite port=init_farthest bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=return        bundle=CTRL

    fps_knn_selection_sort<PRECISION, INT_PREC, N, NPOINT, K, PEs>(
        in_strm,
        grouped_strm,
        init_farthest
    );
}
