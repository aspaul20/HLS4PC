#include <hls_stream.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "hls4pc.hpp"

#define WL      32
#define IWL     16
#define N       512
#define G       256
#define S       16
#define PEs     4
#define DIM     32

bool is_close(float a, float b, float eps = 1e-2f) {
    return std::abs(a - b) < eps;
}

int main() {
    std::ifstream fpt("input_points_S1.txt");
    std::vector<Point<WL,IWL>> pts(N);
    for (int i = 0; i < N; i++) {
        float x,y,z; fpt >> x >> y >> z;
        pts[i].x = x; pts[i].y = y; pts[i].z = z;
    }
    fpt.close();

    std::ifstream fem("input_embeddings_S1.txt");
    std::vector<Embedding<DIM>> embs(N);
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < DIM; d++){
            float v; fem >> v;
            embs[i].data[d] = v;
        }
    }
    fem.close();

    std::ifstream fxyz("grouper_out_xyz_gt.txt");
    std::vector<Point<WL,IWL>> gt_xyz(G);
    for (int g = 0; g < G; g++) {
        float x,y,z;
        fxyz >> x >> y >> z;
        gt_xyz[g].x = x;
        gt_xyz[g].y = y;
        gt_xyz[g].z = z;
    }
    fxyz.close();

    std::ifstream fge("grouper_out_embedding_gt.txt");
    const int W = 2 * DIM;
    std::vector<float> gt_emb(G * S * W);
    for (int g = 0; g < G; g++) {
        for (int s = 0; s < S; s++) {
            for (int w = 0; w < W; w++) {
                fge >> gt_emb[(g*S + s)*W + w];
            }
        }
        std::string tmp; std::getline(fge, tmp); 
    }
    fge.close();

    hls::stream<Point<WL,IWL>>    in_strm;
    hls::stream<Embedding<DIM>>   emb_strm;
    hls::stream<Point<WL,IWL>>    new_xyz_strm;
    hls::stream<Embedding<2*DIM>> new_pts_strm;

    for (auto &p: pts)  in_strm.write(p);
    for (auto &e: embs) emb_strm.write(e);

    fps_knn_grouper<WL,IWL,N,G,S,PEs,DIM>(
        in_strm, emb_strm, new_xyz_strm, new_pts_strm, /*init=*/505
    );
//    urs_knn_grouper<WL,IWL,N,G,S,PEs,DIM>(
//        in_strm, emb_strm, new_xyz_strm, new_pts_strm//, /*init=*/505
//    );

    bool ok = true;
    std::cout << "SampleIdx\tGotXYZ\t\t\tExpXYZ\t\t\tResult\n";
    for (int g = 0; g < G; g++) {
        Point<WL,IWL> o = new_xyz_strm.read();
        float gx = gt_xyz[g].x.to_float();
        float gy = gt_xyz[g].y.to_float();
        float gz = gt_xyz[g].z.to_float();
        float ox = o.x.to_float();
        float oy = o.y.to_float();
        float oz = o.z.to_float();
        bool match = is_close(ox,gx) && is_close(oy,gy) && is_close(oz,gz);
        if (!match) ok = false;
        std::cout
            << g << "\t("
            << ox << "," << oy << "," << oz << ")\t("
            << gx << "," << gy << "," << gz << ")\t\n";
    }

    std::cout << "\nGroup\tNeighbor\tDim\tGot\t\tExp\t\tResult\n";
    for (int g = 0; g < G; g++) {
        for (int s = 0; s < S; s++) {
            Embedding<2*DIM> o = new_pts_strm.read();
            for (int w = 0; w < W; w++) {
                float got = o.data[w].to_float();
                float exp = gt_emb[(g*S + s)*W + w];
                bool match = is_close(got, exp);
                std::cout
                    << g << "\t"
                    << s << "\t"
                    << w << "\t"
                    << got << "\t"
                    << exp << "\t"
                    << "\n";
            }
        }
    }


    return 0;
}
