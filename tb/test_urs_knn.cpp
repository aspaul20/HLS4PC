#include <hls_stream.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "hls4pc.hpp"

#define PRECISION 32
#define INT_PREC 16
bool is_close(float a, float b, float eps = 1e-3f) {
   return std::abs(a - b) < eps;
}

int main() {
   const int N      = 512;
   const int NPOINT = 256;
   const int K      = 16;

   std::ifstream fgt("grouped_point_S1.txt");
   if (!fgt) {
       std::cerr << "cannot open grouped_points_S1.txt\n";
       return 1;
   }

   std::vector<std::tuple<float, float, float>> gt_grouped(NPOINT * K);
   for (int i = 0; i < NPOINT * K; i++) {
       float x, y, z;
       fgt >> x >> y >> z;
       gt_grouped[i] = {x, y, z};
       if ((i + 1) % K == 0) fgt.ignore(10000, '\n'); 
   }
   fgt.close();

   std::ifstream fin("input_points_S1.txt");
   if (!fin) {
       std::cerr << "cannot open input_points_S1.txt\n";
       return 1;
   }

   std::vector<Point<PRECISION, INT_PREC>> buffer(N);
   float xf, yf, zf;
   for (int i = 0; i < N; i++) {
       fin >> xf >> yf >> zf;
       buffer[i].x = xf;
       buffer[i].y = yf;
       buffer[i].z = zf;
   }
   fin.close();

   hls::stream<Point<PRECISION, INT_PREC>> in_strm;
   hls::stream<Point<PRECISION, INT_PREC>> grouped_strm;

   for (int i = 0; i < N; i++) {
       in_strm.write(buffer[i]);
   }

   urs_selection_sort<PRECISION, INT_PREC, N, NPOINT, K, 4>(in_strm, grouped_strm);

   std::cout << "Sample\tNeighbor\tHLS_Out\t\t\tGroundTruth\n";
   for (int i = 0; i < NPOINT * K; i++) {
       Point<PRECISION, INT_PREC> pt = grouped_strm.read();
       float gx = std::get<0>(gt_grouped[i]);
       float gy = std::get<1>(gt_grouped[i]);
       float gz = std::get<2>(gt_grouped[i]);

       std::cout << i / K << "\t" << i % K << "\t("
                 << pt.x.to_float() << ", "
                 << pt.y.to_float() << ", "
                 << pt.z.to_float() << ")"
                 << "\t("
                 << gx << ", "
                 << gy << ", "
                 << gz << ")\n";


   }
   return 0;
}
