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
#define G        256
#define S        16
#define PEs      4
#define D        64

extern "C" void grouper_fps_top(
    hls::stream<Point<PRECISION, INT_PREC>> &in_strm,
    hls::stream<Embedding<D>>              &emb_strm,
    hls::stream<Point<PRECISION, INT_PREC>> &new_xyz_strm,
    hls::stream<Embedding<2 * D>>          &new_pts_strm,
    ap_uint<16>                             init_farthest
) {
    #pragma HLS INTERFACE axis      port=in_strm
    #pragma HLS INTERFACE axis      port=emb_strm
    #pragma HLS INTERFACE axis      port=new_xyz_strm
    #pragma HLS INTERFACE axis      port=new_pts_strm
    #pragma HLS INTERFACE s_axilite port=init_farthest bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=return        bundle=CTRL

    grouper_fps<PRECISION, INT_PREC, N, G, S, PEs, D>(
        in_strm,
        emb_strm,
        new_xyz_strm,
        new_pts_strm,
        init_farthest
    );
}
