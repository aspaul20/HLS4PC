#include <ap_fixed.h>
#include <hls_stream.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <limits>

template<int WL, int IWL>
struct Point {
    ap_fixed<WL, IWL> x, y, z;
};

template<int D>
struct Embedding {
    ap_fixed<32,16> data[D];
};