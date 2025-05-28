#ifndef __SYNTHESIS__
#include <hls_stream.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <iostream>
#include <fstream>
#include <vector>
#endif

#include "hls4pc.hpp"
#define KERNEL_DIM 2
#define IFM_CHANNELS 64
#define IFM_DIM 512
#define STRIDE 2
#define PADDING 0
#define OFM_CHANNELS 64
#define OFM_DIM (IFM_DIM / STRIDE)
#define SIMD_WIDTH 8
#define IN_PREC 8
#define IN_INT_PREC 4
#define OUT_PREC 8
#define OUT_INT_PREC 4

extern "C" void maxpool_top(
    hls::stream<ap_uint<SIMD_WIDTH * IN_PREC> > &in_strm,
    hls::stream<ap_uint<SIMD_WIDTH * OUT_PREC> > &out_strm
) {
    #pragma HLS INTERFACE axis port=in_strm
    #pragma HLS INTERFACE axis port=out_strm
    #pragma HLS INTERFACE s_axilite port=return
    #pragma HLS DATAFLOW

    MaxPool1D<
        KERNEL_DIM,
        IFM_CHANNELS,
        IFM_DIM,
        STRIDE,
        PADDING,
        OFM_CHANNELS,
        OFM_DIM,
        SIMD_WIDTH,
        IN_PREC,
        IN_INT_PREC,
        OUT_PREC,
        OUT_INT_PREC
    >(in_strm, out_strm);
}
