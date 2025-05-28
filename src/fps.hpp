template<
	int WL, 		// Precision
	int IWL, 		// Int Precision
	int N, 			// Total number of points
	int NPOINT, 	// Number of samples
	int UF 			// Number of PEs
>
void fps(
    hls::stream<Point<WL, IWL> > &in_strm,
    hls::stream<ap_uint<16> > &out_strm,
    ap_uint<16> init_farthest
) {
#pragma HLS INTERFACE axis      port=in_strm
#pragma HLS INTERFACE axis      port=out_strm
#pragma HLS INTERFACE s_axilite port=init_farthest bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return        bundle=CTRL

    using acc_t = ap_fixed<32,16>;

    Point<WL, IWL> xyz_local[N];
    acc_t dist_buf[N];

#pragma HLS ARRAY_PARTITION variable=xyz_local cyclic factor=UF
#pragma HLS ARRAY_PARTITION variable=dist_buf  cyclic factor=UF
    ap_fixed<WL, IWL> centroid[3];
    ap_uint<16>      farthest = init_farthest;

    for (int i = 0; i < N; i++) {
    #pragma HLS PIPELINE II=1
        xyz_local[i] = in_strm.read();
    }

    const acc_t INF = std::numeric_limits<acc_t>::max();
    for (int i = 0; i < N; i++) {
    #pragma HLS PIPELINE II=1
        dist_buf[i] = INF;
    }

    for (int s = 0; s < NPOINT; s++) {
        out_strm.write(farthest);

        centroid[0] = xyz_local[farthest].x;
        centroid[1] = xyz_local[farthest].y;
        centroid[2] = xyz_local[farthest].z;

        acc_t maxd = -1;
        ap_uint<16> maxidx = 0;
        for (int i = 0; i < N; i++) {
        #pragma HLS UNROLL factor=UF
            acc_t dx = (acc_t)xyz_local[i].x - centroid[0];
            acc_t dy = (acc_t)xyz_local[i].y - centroid[1];
            acc_t dz = (acc_t)xyz_local[i].z - centroid[2];
            acc_t d2 = dx*dx + dy*dy + dz*dz;
            if (d2 < dist_buf[i]) {
                dist_buf[i] = d2;
            }
            if (dist_buf[i] > maxd) {
                maxd = dist_buf[i];
                maxidx = i;
            }
        }
        farthest = maxidx;
    }
}

template<
	int WL, 		// Precision
	int IWL, 		// Int Precision
	int N, 			// Total number of points
	int NPOINT, 	// Number of samples
	int UF 			// Number of PEs
>
void fps_indexed(
    hls::stream<Point<WL, IWL> > &in_strm,
	hls::stream<Point<WL, IWL> > &out_strm,
    ap_uint<16> init_farthest
) {
#pragma HLS INTERFACE axis      port=in_strm
#pragma HLS INTERFACE axis      port=out_strm
#pragma HLS INTERFACE s_axilite port=init_farthest bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return        bundle=CTRL

    using acc_t = ap_fixed<32,16>;

    Point<WL, IWL> xyz_local[N];
    acc_t dist_buf[N];

#pragma HLS ARRAY_PARTITION variable=xyz_local cyclic factor=UF
#pragma HLS ARRAY_PARTITION variable=dist_buf  cyclic factor=UF
    ap_fixed<WL, IWL> centroid[3];
    ap_uint<16>      farthest = init_farthest;

    for (int i = 0; i < N; i++) {
    #pragma HLS PIPELINE II=1
        xyz_local[i] = in_strm.read();
    }

    const acc_t INF = (acc_t)10000;
    for (int i = 0; i < N; i++) {
    #pragma HLS PIPELINE II=1
        dist_buf[i] = INF;
    }

    for (int s = 0; s < NPOINT; s++) {
    	out_strm.write(xyz_local[farthest]);

        centroid[0] = xyz_local[farthest].x;
        centroid[1] = xyz_local[farthest].y;
        centroid[2] = xyz_local[farthest].z;

        acc_t maxd = -1;
        ap_uint<16> maxidx = 0;
        for (int i = 0; i < N; i++) {
        #pragma HLS UNROLL factor=UF
            acc_t dx = (acc_t)xyz_local[i].x - centroid[0];
            acc_t dy = (acc_t)xyz_local[i].y - centroid[1];
            acc_t dz = (acc_t)xyz_local[i].z - centroid[2];
            acc_t d2 = dx*dx + dy*dy + dz*dz;
            if (d2 < dist_buf[i]) {
                dist_buf[i] = d2;
            }
            if (dist_buf[i] > maxd) {
                maxd = dist_buf[i];
                maxidx = i;
            }
        }
        farthest = maxidx;
    }
}

template<
    int WL,			// Precision
    int IWL,		// Int Precision
    int N,			// Total number of points
    int NPOINT,		// Number of samples
    int K,			// Number of neighbors per sample
    int PEs			// Number of PEs
>
void fps_knn_insertion_sort(
    hls::stream<Point<WL,IWL> >& in_strm,
    hls::stream<Point<WL,IWL> >& grouped_strm,
    ap_uint<16> init_farthest
) {
#pragma HLS INTERFACE axis      port=in_strm
#pragma HLS INTERFACE axis      port=grouped_strm
#pragma HLS INTERFACE s_axilite port=init_farthest bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return        bundle=CTRL

    using acc_t = ap_fixed<32,22>;
    const acc_t INF = (acc_t)1e4;

    Point<WL,IWL> xyz_local[N];
    Point<WL,IWL> sampled[NPOINT];
    acc_t         dist_buf[N];

#pragma HLS ARRAY_PARTITION variable=xyz_local cyclic factor=PEs
#pragma HLS ARRAY_PARTITION variable=dist_buf  cyclic factor=PEs

    for (int i = 0; i < N; i++) {
    #pragma HLS PIPELINE II=1
        xyz_local[i] = in_strm.read();
    }

    for (int i = 0; i < N; i++) {
    #pragma HLS PIPELINE II=1
        dist_buf[i] = INF;
    }

    ap_uint<16> farthest = init_farthest;
    for (int s = 0; s < NPOINT; s++) {
        sampled[s] = xyz_local[farthest];

        acc_t cx = (acc_t)sampled[s].x;
        acc_t cy = (acc_t)sampled[s].y;
        acc_t cz = (acc_t)sampled[s].z;

        acc_t maxd = (acc_t)-1;
        ap_uint<16> maxidx = 0;

        for (int i = 0; i < N; i++) {
        #pragma HLS UNROLL factor=PEs
            acc_t dx = (acc_t)xyz_local[i].x - cx;
            acc_t dy = (acc_t)xyz_local[i].y - cy;
            acc_t dz = (acc_t)xyz_local[i].z - cz;
            acc_t d2 = dx*dx + dy*dy + dz*dz;

            if (d2 < dist_buf[i]) {
                dist_buf[i] = d2;
            }
            if (dist_buf[i] > maxd) {
                maxd   = dist_buf[i];
                maxidx = i;
            }
        }
        farthest = maxidx;
    }

    for (int s = 0; s < NPOINT; s++) {
        acc_t       knn_dists[K];
        ap_uint<16> knn_idx[K];
    #pragma HLS ARRAY_PARTITION variable=knn_dists complete
    #pragma HLS ARRAY_PARTITION variable=knn_idx   complete

        for (int k = 0; k < K; k++) {
        #pragma HLS UNROLL
            knn_dists[k] = INF;
            knn_idx[k]   = 0;
        }

        for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL   factor=PEs
            acc_t dx = (acc_t)xyz_local[i].x - (acc_t)sampled[s].x;
            acc_t dy = (acc_t)xyz_local[i].y - (acc_t)sampled[s].y;
            acc_t dz = (acc_t)xyz_local[i].z - (acc_t)sampled[s].z;
            acc_t d2 = dx*dx + dy*dy + dz*dz;

            for (int k = 0; k < K; k++) {
                if (d2 < knn_dists[k]) {
                    for (int m = K-1; m > k; m--) {
                        knn_dists[m] = knn_dists[m-1];
                        knn_idx[m]   = knn_idx[m-1];
                    }
                    knn_dists[k] = d2;
                    knn_idx[k]   = i;
                    break;
                }
            }
        }

        for (int k = 0; k < K; k++) {
        #pragma HLS PIPELINE II=1
            grouped_strm.write(xyz_local[knn_idx[k]]);
        }
    }
}

template<
    int WL,			// Precision
    int IWL,		// Int Precision
    int N,			// Total number of points
    int NPOINT,		// Number of samples
    int K,			// Number of neighbors per sample
    int PEs			// Number of PEs
>
void fps_knn_selection_sort(
    hls::stream<Point<WL,IWL> >& in_strm,
    hls::stream<Point<WL,IWL> >& grouped_strm,
    ap_uint<16> init_farthest
) {
#pragma HLS INTERFACE axis      port=in_strm
#pragma HLS INTERFACE axis      port=grouped_strm
#pragma HLS INTERFACE s_axilite port=init_farthest bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return        bundle=CTRL

    using acc_t = ap_fixed<32,22>;
    const acc_t INF = (acc_t)1e4;

    Point<WL,IWL> xyz_local[N];
    Point<WL,IWL> sampled[NPOINT];
    acc_t         dist_buf[N];

#pragma HLS ARRAY_PARTITION variable=xyz_local cyclic factor=PEs
#pragma HLS ARRAY_PARTITION variable=dist_buf  cyclic factor=PEs

    for (int i = 0; i < N; i++) {
    #pragma HLS PIPELINE II=1
        xyz_local[i] = in_strm.read();
    }

    for (int i = 0; i < N; i++) {
    #pragma HLS PIPELINE II=1
        dist_buf[i] = INF;
    }

    ap_uint<16> farthest = init_farthest;
    for (int s = 0; s < NPOINT; s++) {
        sampled[s] = xyz_local[farthest];

        acc_t cx = (acc_t)sampled[s].x;
        acc_t cy = (acc_t)sampled[s].y;
        acc_t cz = (acc_t)sampled[s].z;

        acc_t maxd = (acc_t)-1;
        ap_uint<16> maxidx = 0;

        for (int i = 0; i < N; i++) {
        #pragma HLS UNROLL factor=PEs
            acc_t dx = (acc_t)xyz_local[i].x - cx;
            acc_t dy = (acc_t)xyz_local[i].y - cy;
            acc_t dz = (acc_t)xyz_local[i].z - cz;
            acc_t d2 = dx*dx + dy*dy + dz*dz;

            if (d2 < dist_buf[i]) {
                dist_buf[i] = d2;
            }
            if (dist_buf[i] > maxd) {
                maxd   = dist_buf[i];
                maxidx = i;
            }
        }
        farthest = maxidx;
    }

    for (int s = 0; s < NPOINT; s++) {
        acc_t       all_dists[N];
        ap_uint<16> all_indices[N];
    #pragma HLS ARRAY_PARTITION variable=all_dists  cyclic factor=PEs
    #pragma HLS ARRAY_PARTITION variable=all_indices cyclic factor=PEs

        for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
            acc_t dx = (acc_t)xyz_local[i].x - (acc_t)sampled[s].x;
            acc_t dy = (acc_t)xyz_local[i].y - (acc_t)sampled[s].y;
            acc_t dz = (acc_t)xyz_local[i].z - (acc_t)sampled[s].z;
            acc_t d2 = dx*dx + dy*dy + dz*dz;
            all_dists[i] = d2;
            all_indices[i] = i;
        }

        for (int k = 0; k < K; k++) {
            acc_t min_dist = INF;
            ap_uint<16> min_idx = k;
            for (int i = k; i < N; i++) {
                if (all_dists[i] < min_dist) {
                    min_dist = all_dists[i];
                    min_idx = i;
                }
            }

            acc_t tmp_dist = all_dists[k];
            ap_uint<16> tmp_idx = all_indices[k];
            all_dists[k] = all_dists[min_idx];
            all_indices[k] = all_indices[min_idx];
            all_dists[min_idx] = tmp_dist;
            all_indices[min_idx] = tmp_idx;
        }

        for (int k = 0; k < K; k++) {
        #pragma HLS PIPELINE II=1
            grouped_strm.write(xyz_local[all_indices[k]]);
        }
    }
}



template<
    int WL,			// Precision
	int IWL,		// Int Precision
    int N,			// Total number of points
    int G,			// Number of samples/groups
    int S,			// Number of neighbors
    int PEs,		// Number of PEs
    int D			// Embedding channels
>
void grouper_fps(
    hls::stream<Point<WL,IWL> >&  in_strm,
    hls::stream<Embedding<D> >&   emb_strm,
    hls::stream<Point<WL,IWL> >&  new_xyz_strm,
    hls::stream<Embedding<2*D> >& new_pts_strm,
    ap_uint<16>                  init_farthest
) {
#pragma HLS INTERFACE axis      port=in_strm
#pragma HLS INTERFACE axis      port=emb_strm
#pragma HLS INTERFACE axis      port=new_xyz_strm
#pragma HLS INTERFACE axis      port=new_pts_strm
#pragma HLS INTERFACE s_axilite port=init_farthest bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return        bundle=CTRL

    using acc_t = ap_fixed<32,22>;
    const acc_t INF = (acc_t)1e4;

    Point<WL,IWL>    pts[N];
    acc_t            dist_buf[N];
    Point<WL,IWL>    sampled[G];
    ap_uint<16>      fps_idx[G];
#pragma HLS ARRAY_PARTITION variable=pts      cyclic factor=PEs
#pragma HLS ARRAY_PARTITION variable=dist_buf cyclic factor=PEs

    for (int i = 0; i < N; i++) {
    #pragma HLS PIPELINE II=1
        pts[i]      = in_strm.read();
        dist_buf[i] = INF;
    }

    ap_uint<16> far = init_farthest;
    for (int s = 0; s < G; s++) {
        sampled[s] = pts[far];
        fps_idx[s] = far;

        acc_t cx = (acc_t)sampled[s].x;
        acc_t cy = (acc_t)sampled[s].y;
        acc_t cz = (acc_t)sampled[s].z;
        acc_t maxd = (acc_t)-1; ap_uint<16> midx=0;

        for (int i = 0; i < N; i++) {
        #pragma HLS UNROLL factor=PEs
            acc_t dx = (acc_t)pts[i].x - cx;
            acc_t dy = (acc_t)pts[i].y - cy;
            acc_t dz = (acc_t)pts[i].z - cz;
            acc_t d2 = dx*dx + dy*dy + dz*dz;
            if (d2 < dist_buf[i]) dist_buf[i] = d2;
            if (dist_buf[i] > maxd) {
                maxd = dist_buf[i];
                midx = i;
            }
        }
        far = midx;
    }

    for (int s = 0; s < G; s++) {
    #pragma HLS PIPELINE II=1
        new_xyz_strm.write(sampled[s]);
    }

    Embedding<D> emb_buf[N];
#pragma HLS ARRAY_PARTITION variable=emb_buf complete dim=2
    for (int i = 0; i < N; i++) {
    #pragma HLS PIPELINE II=1
        emb_buf[i] = emb_strm.read();
    }

    for (int s = 0; s < G; s++) {
        acc_t       dists[N];
        ap_uint<16> idxs[N];
    #pragma HLS ARRAY_PARTITION variable=dists cyclic factor=PEs
    #pragma HLS ARRAY_PARTITION variable=idxs  cyclic factor=PEs

        // compute distances
        for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
            acc_t dx = (acc_t)pts[i].x - (acc_t)sampled[s].x;
            acc_t dy = (acc_t)pts[i].y - (acc_t)sampled[s].y;
            acc_t dz = (acc_t)pts[i].z - (acc_t)sampled[s].z;
            dists[i] = dx*dx + dy*dy + dz*dz;
            idxs[i]  = i;
        }

        for (int k = 0; k < S; k++) {
            acc_t      best = INF;
            ap_uint<16> bidx = k;
            for (int i = k; i < N; i++) {
                if (dists[i] < best) {
                    best = dists[i];
                    bidx = i;
                }
            }
            float temp_dist = dists[k];
			dists[k] = dists[bidx];
			dists[bidx] = temp_dist;

			int temp_idx = idxs[k];
			idxs[k] = idxs[bidx];
			idxs[bidx] = temp_idx;
        }

        Embedding<D> fps_emb = emb_buf[fps_idx[s]];

        for (int k = 0; k < S; k++) {
        #pragma HLS PIPELINE II=1
            Embedding<2*D> out;
            Embedding<D>    grp = emb_buf[idxs[k]];
            for (int d = 0; d < D; d++) {
                out.data[d]     = grp.data[d];
                out.data[D + d] = fps_emb.data[d];
            }
            new_pts_strm.write(out);
        }
    }
}
