#ifndef __SYNTHESIS__
#include <hls_stream.h>
#include <ap_int.h>
#endif
#include "hls4pc.hpp"
#define WL           16
#define IWL           8
#define N           1024
#define NPOINT       256
#define K             8
#define PEs           4
#define SEED_WIDTH   32

extern "C" void urs_knn_selection_sort_top(
    hls::stream<Point<WL, IWL> >& in_strm,
    hls::stream<Point<WL, IWL> >& grouped_strm
) {
#pragma HLS INTERFACE axis      port=in_strm
#pragma HLS INTERFACE axis      port=grouped_strm
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    urs_knn_selection_sort<
        WL,
        IWL,
        N,
        NPOINT,
        K,
        PEs,
        SEED_WIDTH
    >(in_strm, grouped_strm);
}
