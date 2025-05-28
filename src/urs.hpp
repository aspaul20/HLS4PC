template<
	int MAX_N, 			// Total number of points
	int MAX_NPOINT,		// Number of samples
	int INDEX_WIDTH,	// Precision for output samples
	int SEED_WIDTH		// Precision for seed
>
void urs(
    hls::stream<ap_uint<INDEX_WIDTH> > &centroid_stream
    )
{
#pragma HLS INTERFACE axis port=centroid_stream
#pragma HLS INTERFACE s_axilite port=return  bundle=control

    ap_uint<SEED_WIDTH> seed = 42;

    ap_uint<INDEX_WIDTH> idx_array[MAX_N];
#pragma HLS ARRAY_PARTITION variable=idx_array block factor=8 dim=1
    init_loop: for (int i = 0; i < MAX_N; i++) {
#pragma HLS PIPELINE II=1
        idx_array[i] = i;
    }

    shuffle_loop: for (int i = 0; i < MAX_N; i++) {
#pragma HLS PIPELINE II=1
        bool lfsr_bit = (seed[0] ^ seed[2] ^ seed[3] ^ seed[5]);
        seed = (seed >> 1) | (ap_uint<SEED_WIDTH>(lfsr_bit) << (SEED_WIDTH - 1));
        ap_uint<INDEX_WIDTH> rand_offset = seed % (MAX_N - i);

        ap_uint<INDEX_WIDTH> temp = idx_array[i];
        idx_array[i] = idx_array[i + rand_offset];
        idx_array[i + rand_offset] = temp;
    }

    output_loop: for (int i = 0; i < MAX_NPOINT; i++) {
#pragma HLS PIPELINE II=1
        centroid_stream.write(idx_array[i]);
    }
}

template<
    int WL,				// Precision
	int IWL,			// Int Precision
    int MAX_N,			// Total number of points
	int MAX_NPOINT,		// Number of samples
    int INDEX_WIDTH,	// Precision for output samples
    int SEED_WIDTH		// Precision for seed
>
void urs_indexed(
    hls::stream<Point<WL, IWL>> &in_strm,
    hls::stream<Point<WL, IWL>> &out_strm
) {
#pragma HLS INTERFACE axis port=in_strm
#pragma HLS INTERFACE axis port=out_strm
#pragma HLS INTERFACE s_axilite port=return  bundle=CTRL

    ap_uint<SEED_WIDTH> seed = 42;

    Point<WL, IWL> point_buf[MAX_N];
    ap_uint<INDEX_WIDTH> idx_array[MAX_N];
#pragma HLS ARRAY_PARTITION variable=point_buf block factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=idx_array block factor=8 dim=1

    read_loop: for (int i = 0; i < MAX_N; i++) {
    #pragma HLS PIPELINE II=1
        point_buf[i] = in_strm.read();
        idx_array[i] = i;
    }

    shuffle_loop: for (int i = 0; i < MAX_N; i++) {
    #pragma HLS PIPELINE II=1
        bool lfsr_bit = seed[0] ^ seed[2] ^ seed[3] ^ seed[5];
        seed = (seed >> 1) | (ap_uint<SEED_WIDTH>(lfsr_bit) << (SEED_WIDTH - 1));
        ap_uint<INDEX_WIDTH> rand_offset = seed % (MAX_N - i);

        ap_uint<INDEX_WIDTH> temp = idx_array[i];
        idx_array[i] = idx_array[i + rand_offset];
        idx_array[i + rand_offset] = temp;
    }

    output_loop: for (int i = 0; i < MAX_NPOINT; i++) {
    #pragma HLS PIPELINE II=1
        ap_uint<INDEX_WIDTH> idx = idx_array[i];
        out_strm.write(point_buf[idx]);
    }
}


template<
    int WL,				// Precision
    int IWL,			// Int Precision
    int N,				// Total number of points
    int NPOINT,			// Number of samples
    int K,				// Number of neighbors
    int PEs,     		// Number of PEs
    int SEED_WIDTH		// Precision for seed
>
void urs_knn_insertion_sort(
    hls::stream<Point<WL,IWL>>& in_strm,
    hls::stream<Point<WL,IWL>>& grouped_strm
) {
#pragma HLS INTERFACE axis      port=in_strm
#pragma HLS INTERFACE axis      port=grouped_strm
#pragma HLS INTERFACE s_axilite port=return        bundle=CTRL

    using acc_t = ap_fixed<32,22>;
    const acc_t INF = (acc_t)1e4;

    Point<WL,IWL> xyz_local[N];
#pragma HLS ARRAY_PARTITION variable=xyz_local cyclic factor=PEs
    for (int i = 0; i < N; i++) {
    #pragma HLS PIPELINE II=1
        xyz_local[i] = in_strm.read();
    }

    ap_uint<16> idx_array[N];
#pragma HLS ARRAY_PARTITION variable=idx_array block factor=8 dim=1
    ap_uint<SEED_WIDTH> seed = 42;

    for (int i = 0; i < N; i++) {
    #pragma HLS PIPELINE II=1
        idx_array[i] = i;
    }

    for (int i = 0; i < N - 1; i++) {
    #pragma HLS PIPELINE II=1
        bool lfsr_bit = seed[0] ^ seed[2] ^ seed[3] ^ seed[5];
        seed = (seed >> 1) | (ap_uint<SEED_WIDTH>(lfsr_bit) << (SEED_WIDTH - 1));
        ap_uint<16> rand_offset = seed % (N - i);

        ap_uint<16> temp = idx_array[i];
        idx_array[i] = idx_array[i + rand_offset];
        idx_array[i + rand_offset] = temp;
    }

    Point<WL,IWL> sampled[NPOINT];
    for (int i = 0; i < NPOINT; i++) {
    #pragma HLS PIPELINE II=1
        sampled[i] = xyz_local[idx_array[i]];
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
                    for (int m = K - 1; m > k; m--) {
                        knn_dists[m] = knn_dists[m - 1];
                        knn_idx[m]   = knn_idx[m - 1];
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
    int WL,				// Precision
    int IWL,			// Int Precision
    int N,				// Total number of points
    int NPOINT,			// Number of samples
    int K,				// Number of neighbors
    int PEs,     		// Number of PEs
    int SEED_WIDTH		// Precision for seed
>
void urs_knn_selection_sort(
    hls::stream<Point<WL,IWL>>& in_strm,
    hls::stream<Point<WL,IWL>>& grouped_strm
) {
#pragma HLS INTERFACE axis      port=in_strm
#pragma HLS INTERFACE axis      port=grouped_strm
#pragma HLS INTERFACE s_axilite port=return        bundle=CTRL

    using acc_t = ap_fixed<32,22>;
    const acc_t INF = (acc_t)1e4;

    Point<WL,IWL> xyz_local[N];
    ap_uint<16>   sampled_idx[NPOINT];

    ap_uint<SEED_WIDTH> seed = 42;

    for (int i = 0; i < N; i++) {
    #pragma HLS PIPELINE II=1
        xyz_local[i] = in_strm.read();
    }


    ap_uint<16> idx_array[N];
#pragma HLS ARRAY_PARTITION variable=idx_array block factor=8 dim=1

    for (int i = 0; i < N; i++) {
    #pragma HLS PIPELINE II=1
        idx_array[i] = i;
    }

    for (int i = 0; i < N - 1; i++) {
    #pragma HLS PIPELINE II=1
        bool lfsr_bit = (seed[0] ^ seed[2] ^ seed[3] ^ seed[5]);
        seed = (seed >> 1) | (ap_uint<SEED_WIDTH>(lfsr_bit) << (SEED_WIDTH - 1));
        ap_uint<16> rand_offset = seed % (N - i);

        ap_uint<16> temp = idx_array[i];
        idx_array[i] = idx_array[i + rand_offset];
        idx_array[i + rand_offset] = temp;
    }

    for (int i = 0; i < NPOINT; i++) {
    #pragma HLS PIPELINE II=1
        sampled_idx[i] = idx_array[i];
    }


    for (int s = 0; s < NPOINT; s++) {
        Point<WL,IWL> center = xyz_local[sampled_idx[s]];
        acc_t       all_dists[N];
        ap_uint<16> all_indices[N];
    #pragma HLS ARRAY_PARTITION variable=all_dists  block factor=PEs dim=1
    #pragma HLS ARRAY_PARTITION variable=all_indices block factor=PEs dim=1

        for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
            acc_t dx = (acc_t)xyz_local[i].x - (acc_t)center.x;
            acc_t dy = (acc_t)xyz_local[i].y - (acc_t)center.y;
            acc_t dz = (acc_t)xyz_local[i].z - (acc_t)center.z;
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
    int WL,				// Precision
	int IWL,			// Int Precision
    int N,				// Total number of points
    int G,				// Number of samples/groups
    int S,				// Number of neighbors
    int PEs,			// Number of PEs
    int D,				// Embedding channels
    int SEED_WIDTH		// Precision for seed
>
void grouper_urs(
    hls::stream<Point<WL,IWL>>&  in_strm,
    hls::stream<Embedding<D>>&   emb_strm,
    hls::stream<Point<WL,IWL>>&  new_xyz_strm,
    hls::stream<Embedding<2*D>>& new_pts_strm
) {
#pragma HLS INTERFACE axis      port=in_strm
#pragma HLS INTERFACE axis      port=emb_strm
#pragma HLS INTERFACE axis      port=new_xyz_strm
#pragma HLS INTERFACE axis      port=new_pts_strm
#pragma HLS INTERFACE s_axilite port=return        bundle=CTRL

    using acc_t = ap_fixed<32,22>;
    const acc_t INF = (acc_t)1e4;

    Point<WL,IWL> pts[N];
    Embedding<D>  emb_buf[N];
    ap_uint<16>   sampled_idx[G];

#pragma HLS ARRAY_PARTITION variable=pts cyclic factor=PEs
#pragma HLS ARRAY_PARTITION variable=emb_buf cyclic factor=PEs

    for (int i = 0; i < N; i++) {
#pragma HLS UNROLL factor=PEs
        pts[i] = in_strm.read();
        emb_buf[i] = emb_strm.read();
    }

    ap_uint<16> idx_array[N];
#pragma HLS ARRAY_PARTITION variable=idx_array complete
    for (int i = 0; i < N; i++) {
#pragma HLS UNROLL factor=PEs
        idx_array[i] = i;
    }

    ap_uint<SEED_WIDTH> seed = 42;
    for (int i = 0; i < N - 1; i++) {
#pragma HLS UNROLL factor=PEs
        bool lfsr_bit = (seed[0] ^ seed[2] ^ seed[3] ^ seed[5]);
        seed = (seed >> 1) | (ap_uint<SEED_WIDTH>(lfsr_bit) << (SEED_WIDTH - 1));
        ap_uint<16> rand_offset = seed % (N - i);
        int temp = idx_array[i];
        idx_array[i] = idx_array[i + rand_offset];
        idx_array[i + rand_offset] = temp;
    }

    for (int i = 0; i < G; i++) {
#pragma HLS UNROLL factor=PEs
        sampled_idx[i] = idx_array[i];
    }

    for (int s = 0; s < G; s++) {
        Point<WL,IWL> center = pts[sampled_idx[s]];
        Embedding<D>  center_emb = emb_buf[sampled_idx[s]];

        acc_t dists[N];
        ap_uint<16> idxs[N];
#pragma HLS ARRAY_PARTITION variable=dists block factor=PEs dim=1
#pragma HLS ARRAY_PARTITION variable=idxs  block factor=PEs dim=1

        for (int i = 0; i < N; i++) {
#pragma HLS UNROLL factor=PEs
            acc_t dx = (acc_t)pts[i].x - (acc_t)center.x;
            acc_t dy = (acc_t)pts[i].y - (acc_t)center.y;
            acc_t dz = (acc_t)pts[i].z - (acc_t)center.z;
            dists[i] = dx*dx + dy*dy + dz*dz;
            idxs[i]  = i;
        }

        for (int k = 0; k < S; k++) {
            acc_t best = INF;
            ap_uint<16> bidx = k;
            for (int i = k; i < N; i++) {
#pragma HLS UNROLL factor=PEs
                if (dists[i] < best) {
                    best = dists[i];
                    bidx = i;
                }
            }
            auto temp_dist = dists[k];
            dists[k] = dists[bidx];
            dists[bidx] = temp_dist;

            auto temp_idx = idxs[k];
            idxs[k] = idxs[bidx];
            idxs[bidx] = temp_idx;
        }

        new_xyz_strm.write(center);

        for (int k = 0; k < S; k++) {
#pragma HLS UNROLL factor=PEs
            Embedding<2*D> out;
            Embedding<D> grp = emb_buf[idxs[k]];
            for (int d = 0; d < D; d++) {
#pragma HLS UNROLL
                out.data[d]     = grp.data[d];
                out.data[D + d] = center_emb.data[d];
            }
            new_pts_strm.write(out);
        }
    }
}
