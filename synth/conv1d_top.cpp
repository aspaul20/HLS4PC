#ifndef __SYNTHESIS__
#include <hls_stream.h>
#include <ap_int.h>
#endif

#include "hls4pc.hpp"
#define MAX_KERNEL_DIM     3
#define MAX_IFM_CHANNELS   64
#define MAX_IFM_DIM        4096
#define MAX_OFM_CHANNELS   64
#define MAX_OFM_DIM        4094
#define PE_COUNT           4
#define SIMD_WIDTH         1
#define INPUT_PRECISION    8
#define INPUT_INT_PREC     1
#define WEIGHT_PRECISION   8
#define WEIGHT_INT_PREC    1
#define BIAS_PRECISION     8
#define BIAS_INT_PREC      1
#define MUL_PRECISION      16
#define MUL_INT_PREC       4
#define ACC_PRECISION      32
#define ACC_INT_PREC       8
#define OUTPUT_PRECISION   8
#define OUTPUT_INT_PREC    1

extern "C" void conv1d_top(
    hls::stream<ap_uint<SIMD_WIDTH * INPUT_PRECISION>> &in,
    hls::stream<ap_uint<PE_COUNT * OUTPUT_PRECISION>> &out,
    const ap_uint<WEIGHT_PRECISION> weightMem[PE_COUNT][SIMD_WIDTH][MAX_KERNEL_DIM * MAX_IFM_CHANNELS * MAX_OFM_CHANNELS / (SIMD_WIDTH * PE_COUNT)],
    const ap_uint<BIAS_PRECISION> biasMem[PE_COUNT][MAX_OFM_CHANNELS / PE_COUNT]
) {
#pragma HLS INTERFACE axis      port=in
#pragma HLS INTERFACE axis      port=out
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE ap_memory port=weightMem
#pragma HLS INTERFACE ap_memory port=biasMem

#pragma HLS DATAFLOW

    hls::stream<ap_uint<SIMD_WIDTH * INPUT_PRECISION>> buffer_to_mac_stream;
#pragma HLS STREAM variable=buffer_to_mac_stream depth=64

    Conv1DBuffer<
        MAX_KERNEL_DIM,
        MAX_IFM_CHANNELS,
        MAX_IFM_DIM,
        1, // stride
        MAX_OFM_CHANNELS,
        MAX_OFM_DIM,
        PE_COUNT,
        SIMD_WIDTH,
        INPUT_PRECISION,
        INPUT_INT_PREC
    >(in, buffer_to_mac_stream);

    Conv1DMac<
        MAX_KERNEL_DIM,
        MAX_IFM_CHANNELS,
        MAX_IFM_DIM,
        1,
        0,
        MAX_OFM_CHANNELS,
        MAX_OFM_DIM,
        PE_COUNT,
        SIMD_WIDTH,
        BIAS_PRECISION,
        BIAS_INT_PREC,
        WEIGHT_PRECISION,
        WEIGHT_INT_PREC,
        INPUT_PRECISION,
        INPUT_INT_PREC,
        MUL_PRECISION,
        MUL_INT_PREC,
        ACC_PRECISION,
        ACC_INT_PREC,
        OUTPUT_PRECISION,
        OUTPUT_INT_PREC
    >(buffer_to_mac_stream, out, weightMem, biasMem);
}
