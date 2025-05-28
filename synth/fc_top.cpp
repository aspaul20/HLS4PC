#ifndef __SYNTHESIS__
#include <hls_stream.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <iostream>
#include <fstream>
#endif

#include "hls4pc.hpp"
#define INPUTS 512
#define NEURONS 256
#define PECOUNT 8
#define SIMD 8
#define BIAS_PREC 16
#define BIAS_INT_PREC 4
#define WEIGHT_PREC 8
#define WEIGHT_INT_PREC 4
#define IN_PREC 8
#define IN_INT_PREC 4
#define MUL_PREC 16
#define MUL_INT_PREC 6
#define ACC_PREC 24
#define ACC_INT_PREC 8
#define OUT_PREC 8
#define OUT_INT_PREC 4

extern "C" void fc_mac_top(
    hls::stream<ap_uint<SIMD * IN_PREC> > &in_strm,
    hls::stream<ap_uint<PECOUNT * OUT_PREC> > &out_strm,
    const ap_uint<WEIGHT_PREC> weightMem[PECOUNT][SIMD][INPUTS * NEURONS / (SIMD * PECOUNT)],
    const ap_uint<BIAS_PREC> biasMem[PECOUNT][NEURONS / PECOUNT]
) {
    #pragma HLS INTERFACE axis port=in_strm
    #pragma HLS INTERFACE axis port=out_strm
	#pragma HLS INTERFACE ap_memory port=weightMem
	#pragma HLS INTERFACE ap_memory port=biasMem
    #pragma HLS INTERFACE s_axilite port=return
    #pragma HLS DATAFLOW

    FCMac<
        INPUTS, NEURONS,
        PECOUNT, SIMD,
        BIAS_PREC, BIAS_INT_PREC,
        WEIGHT_PREC, WEIGHT_INT_PREC,
        IN_PREC, IN_INT_PREC,
        MUL_PREC, MUL_INT_PREC,
        ACC_PREC, ACC_INT_PREC,
        OUT_PREC, OUT_INT_PREC
    >(in_strm, out_strm, weightMem, biasMem);
}
