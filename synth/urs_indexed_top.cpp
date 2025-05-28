#ifndef __SYNTHESIS__
#include <hls_stream.h>
#include <ap_int.h>
#endif
#include "hls4pc.hpp"

#define WL            16
#define IWL           8
#define MAX_N         1024
#define MAX_NPOINT    256
#define INDEX_WIDTH   16
#define SEED_WIDTH    32

extern "C" void urs_indexed_top(
    hls::stream<Point<WL, IWL>> &in_strm,
    hls::stream<Point<WL, IWL>> &out_strm
) {
#pragma HLS INTERFACE axis      port=in_strm
#pragma HLS INTERFACE axis      port=out_strm
#pragma HLS INTERFACE s_axilite port=return   bundle=CTRL

    urs_indexed<
        WL,
        IWL,
        MAX_N,
        MAX_NPOINT,
        INDEX_WIDTH,
        SEED_WIDTH
    >(in_strm, out_strm);
}
