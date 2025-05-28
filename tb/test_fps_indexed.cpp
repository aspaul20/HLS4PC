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

   std::ifstream fgt("indexed_points_S1.txt");
   if (!fgt) {
       std::cerr << "cannot open indexed_points.txt\n";
       return 1;
   }

   std::vector<std::tuple<float, float, float>> gt_points(NPOINT);
   for (int i = 0; i < NPOINT; i++) {
       float x, y, z;
       fgt >> x >> y >> z;
       gt_points[i] = {x, y, z};
   }
   fgt.close();

   std::vector<Point<PRECISION, INT_PREC>> buffer(N);
   std::ifstream fin("input_points_S1.txt");
   if (!fin) {
       std::cerr << "cannot open input_points_S1.txt\n";
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

   hls::stream<Point<PRECISION, INT_PREC>> in_strm;
   hls::stream<Point<PRECISION, INT_PREC>> out_strm;
   for (int i = 0; i < N; i++) {
       in_strm.write(buffer[i]);
   }

   farthest_point_sample_indexed<PRECISION, INT_PREC, N, NPOINT, 4>(in_strm, out_strm, 505);

   int correct = 0;
   std::cout << "Sample\tHLS_Out\t\t\tGroundTruth\n";
   for (int i = 0; i < NPOINT; i++) {
       Point<PRECISION, INT_PREC> pt = out_strm.read();
       float gx = std::get<0>(gt_points[i]);
       float gy = std::get<1>(gt_points[i]);
       float gz = std::get<2>(gt_points[i]);

       std::cout << i << "\t("
                 << pt.x.to_float() << ", "
                 << pt.y.to_float() << ", "
                 << pt.z.to_float() << ")"
                 << "\t("
                 << gx << ", "
                 << gy << ", "
                 << gz << ")\n";

       if (is_close(pt.x.to_float(), gx) &&
           is_close(pt.y.to_float(), gy) &&
           is_close(pt.z.to_float(), gz)) {
           correct++;
       }
   }

   float accuracy = (float)correct / NPOINT;
   std::cout << "Accuracy: " << accuracy * 100 << "% (" << correct << "/" << NPOINT << " matched)\n";

   return 0;
}

