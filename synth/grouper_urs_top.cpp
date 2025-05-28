#ifndef __SYNTHESIS__
#include <hls_stream.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <iostream>
#include <fstream>
#include <vector>
#endif

#include "hls4pc.hpp"
#define WL 32
#define IWL 16
#define N 512
#define G 256
#define S 16
#define PEs 8
#define D 64
#define SEED_WIDTH 16

extern "C" {
void grouper_urs_top(
    hls::stream<Point<WL, IWL>>&  in_strm,
    hls::stream<Embedding<D>>&    emb_strm,
    hls::stream<Point<WL, IWL>>&  new_xyz_strm,
    hls::stream<Embedding<2*D>>&  new_pts_strm
) {
#pragma HLS INTERFACE axis      port=in_strm
#pragma HLS INTERFACE axis      port=emb_strm
#pragma HLS INTERFACE axis      port=new_xyz_strm
#pragma HLS INTERFACE axis      port=new_pts_strm
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    grouper_urs<WL, IWL, N, G, S, PEs, D, SEED_WIDTH>(
        in_strm, emb_strm, new_xyz_strm, new_pts_strm
    );
}
}
