#include <hls_stream.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "hls4pc.hpp"

#define PRECISION 32
#define INT_PREC 16

int main() {
   const int N      = 512;
   const int NPOINT = 256;

   std::ifstream fgt("sampled_points_S1.txt");
   if (!fgt) {
       std::cerr << "cannot open sampled_points.txt\n";
       return 1;
   }
   std::vector<int> ground(NPOINT);
   for (int i = 0; i < NPOINT; i++) {
       fgt >> ground[i];
   }
   fgt.close();

   std::vector<Point<PRECISION,INT_PREC>> buffer(N);
   std::ifstream fin("input_points_S1.txt");
   if (!fin) {
       std::cerr << "cannot open input_points.txt\n";
       return 1;
   }
   float xf, yf, zf;
   for (int i = 0; i < N; i++) {
       fin >> xf >> yf >> zf;
       buffer[i].x = xf;
       buffer[i].y = yf;
       buffer[i].z = zf;
   }
   fin.close();

   hls::stream<Point<PRECISION,INT_PREC>> in_strm;
   hls::stream<ap_uint<16>> out_strm;
   for (int i = 0; i < N; i++) {
       in_strm.write(buffer[i]);
   }

   farthest_point_sample<PRECISION, INT_PREC, N, NPOINT, 4>(in_strm, out_strm, ground[0]);

   int correct = 0;
   std::cout << "Sample\tHLS_Out\tGroundTruth\n";
   for (int i = 0; i < NPOINT; i++) {
       int hls_idx = out_strm.read().to_uint();
       std::cout << i << "\t" << hls_idx << "\t" << ground[i] << "\n";
       if (hls_idx == ground[i]) {
           correct++;
       }
   }

   float accuracy = (float)correct / NPOINT;
   std::cout << "Accuracy: " << accuracy * 100 << "% (" << correct << "/" << NPOINT << " matched)\n";

   return 0;
}
