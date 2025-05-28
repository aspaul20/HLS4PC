#ifndef __SYNTHESIS__
#include <hls_stream.h>
#include <ap_int.h>
#endif
#include "hls4pc.hpp"
#define MAX_N        1024
#define MAX_NPOINT   256
#define INDEX_WIDTH  16
#define SEED_WIDTH   32

extern "C" void urs_top(
    hls::stream<ap_uint<INDEX_WIDTH>> &centroid_stream
) {
    #pragma HLS INTERFACE axis      port=centroid_stream
    #pragma HLS INTERFACE s_axilite port=return   bundle=control

    urs<MAX_N, MAX_NPOINT, INDEX_WIDTH, SEED_WIDTH>(
        centroid_stream
    );
}
