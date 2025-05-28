#include "types.hpp"
#include "fps.hpp"
#include "urs.hpp"
#include "nn.hpp"

// Furthest point sampling

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
);

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
);

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
);

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
);

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
);



// Uniform Random Sampling 

template<
	int MAX_N, 			// Total number of points
	int MAX_NPOINT,		// Number of samples
	int INDEX_WIDTH,	// Precision for output samples
	int SEED_WIDTH		// Precision for seed
>
void urs(
    hls::stream<ap_uint<INDEX_WIDTH> > &centroid_stream
);

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
);

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
    hls::stream<Point<WL,IWL> >& in_strm,
    hls::stream<Point<WL,IWL> >& grouped_strm
);

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
    hls::stream<Point<WL,IWL> >& in_strm,
    hls::stream<Point<WL,IWL> >& grouped_strm
);

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
    hls::stream<Point<WL,IWL> >&  in_strm,
    hls::stream<Embedding<D> >&   emb_strm,
    hls::stream<Point<WL,IWL> >&  new_xyz_strm,
    hls::stream<Embedding<2*D> >& new_pts_strm
);

template<
     short unsigned int KernelDim, 			// Conv kernel dim
     short unsigned int IFMChannels,		// Max number of input feature maps
     short unsigned int IFMDim,				// Max width of input feature map
     short unsigned int Stride,				// Stride of kernel
     short unsigned int OFMChannels,		// Max number of output feature maps
     short unsigned int OFMDim,				// Max width of output feature map
     short unsigned int PECount,			// Number of PEs
     short unsigned int SIMDWidth,			// Number of SIMD lanes
     short unsigned int Precision,			// Precision
     short unsigned int IntPrecision		// Int Precision
>
void Conv1DBuffer(hls::stream<ap_uint<SIMDWidth * Precision> >& in,
                      hls::stream<ap_uint<SIMDWidth * Precision> >& out);

template<
		short unsigned int KernelDim, 			// Conv kernel dim
		short unsigned int IFMChannels,			// Max number of input feature maps
		short unsigned int IFMDim,				// Max width of input feature map
		short unsigned int Stride,				// Stride of kernel
		short unsigned int Padding,				// Padding alongside kernel
		short unsigned int OFMChannels,			// Max number of output feature maps
		short unsigned int OFMDim,				// Max width of output feature map
		short unsigned int PECount,				// Number of PEs
		short unsigned int SIMDWidth,			// Number of SIMD lanes
		short unsigned int BiasPrecision,		// Total bias precision
		short unsigned int BiasIntPrecision,	// Bias int precision
		short unsigned int WeightsPrecision,	// Total weight precision
		short unsigned int WeightsIntPrecision, // Weight int precision
		short unsigned int InputPrecision,		// Total input precision
		short unsigned int InputIntPrecision,	// Input int precision
		short unsigned int MulPrecision,		// Total multiplier precision
		short unsigned int MulIntPrecision,		// Multiplier int pprecision
		short unsigned int AccPrecision,		// Total accumulator precision
		short unsigned int AccIntPrecision,		// Accumulator int precision
		short unsigned int OutputPrecision,		// Total output precision
		short unsigned int OutputIntPrecision	// Output int precision
>
void Conv1DMac(hls::stream<ap_uint<SIMDWidth * InputPrecision> > & in,
			   hls::stream<ap_uint<PECount * OutputPrecision> > & out,
			   const ap_uint<WeightsPrecision> weightMem[PECount][SIMDWidth][KernelDim * IFMChannels * OFMChannels / (SIMDWidth * PECount)],
			   const ap_uint<BiasPrecision> biasMem [PECount][OFMChannels / PECount]);

template<
		 short unsigned int KernelDim,				// Conv kernel dim
		 short unsigned int IFMChannels,			// Max number of input feature maps
		 short unsigned int IFMDim,					// Max width of input feature map
		 short unsigned int Stride,					// Stride of kernel
		 short unsigned int OFMChannels,			// Max number of output feature maps
		 short unsigned int OFMDim,               	// Max width of output feature map
		 short unsigned int PECount,				// Number of PEs
		 short unsigned int SIMDWidth,              // Number of SIMD lanes
		 short unsigned int Precision,         		// Precisions for the input/output activation
		 short unsigned int IntPrecision      		// Input/Output activation int bitwidth
 >
void Conv1D_pointwise_buffer(hls::stream<ap_uint<SIMDWidth * Precision> > & in,
				  hls::stream<ap_uint<SIMDWidth * Precision> > & out);

template<
		short unsigned int KernelDim,        		// Conv kernel dim
		short unsigned int IFMChannels,				// Max number of input feature maps
		short unsigned int IFMDim,               	// Max width of input feature map
		short unsigned int Stride,					// Stride of kernel
		short unsigned int Padding,					// Padding alongside kernel
		short unsigned int OFMChannels,          	// Max number of output feature maps
		short unsigned int OFMDim,               	// Max width of output feature map
		short unsigned int PECount,                 // Number of PEs
		short unsigned int SIMDWidth,               // Number of SIMD lanes
		short unsigned int BiasPrecision,        	// Precisions for the bias
		short unsigned int BiasIntPrecision,     	// Bias int bitwidth
		short unsigned int WeightsPrecision,        // Precisions for the weight
		short unsigned int WeightsIntPrecision,     // Weight int bitwidth
		short unsigned int InputPrecision,          // Precisions for the input activation
		short unsigned int InputIntPrecision,       // Input activation int bitwidth
		short unsigned int MulPrecision,            // Precision for the result of multiplication
		short unsigned int MulIntPrecision,         // Multiplication int bitwidth
		short unsigned int AccPrecision,            // Precision for the result of accumulation
		short unsigned int AccIntPrecision,         // Accumulation int bitwidth
		short unsigned int OutputPrecision,			// Precisions for the output activation
		short unsigned int OutputIntPrecision		// Output int precision
>
void Conv1D_pointwise_mac(hls::stream<ap_uint<SIMDWidth * InputPrecision> > & in,
			   hls::stream<ap_uint<PECount * OutputPrecision> > & out,
			   const ap_uint<WeightsPrecision> weightMem[PECount][SIMDWidth][KernelDim * IFMChannels * OFMChannels / (SIMDWidth * PECount)],
			   const ap_uint<BiasPrecision> biasMem [PECount][OFMChannels / PECount]);


template<
		short unsigned int Inputs,					// Number of inputs: dim * ch
		short unsigned int Neurons,               	// Number of units
		short unsigned int PECount,                 // Number of PEs
		short unsigned int SIMDWidth,               // Number of SIMD lanes
		short unsigned int BiasPrecision,        	// Precisions for the bias
		short unsigned int BiasIntPrecision,     	// Bias int bitwidth
		short unsigned int WeightsPrecision,        // Precisions for the weight
		short unsigned int WeightsIntPrecision,     // Weight int bitwidth
		short unsigned int InputPrecision,          // Precisions for the input activation
		short unsigned int InputIntPrecision,       // Input activation int bitwidth
		short unsigned int MulPrecision,            // Precision for the result of multiplication
		short unsigned int MulIntPrecision,         // Multiplication int bitwidth
		short unsigned int AccPrecision,            // Precision for the result of accumulation
		short unsigned int AccIntPrecision,         // Accumulation int bitwidth
		short unsigned int OutputPrecision,			// Precisions for the output activation
		short unsigned int OutputIntPrecision
>
void FCMac(hls::stream<ap_uint<SIMDWidth * InputPrecision> > & in,
		   hls::stream<ap_uint<PECount * OutputPrecision> > & out,
		   const ap_uint<WeightsPrecision> weightMem[PECount][SIMDWidth][Inputs * Neurons / (SIMDWidth * PECount)],
		   const ap_uint<BiasPrecision> biasMem [PECount][Neurons / PECount]);

template<
		 short unsigned int IFMChannels,			// Number of input feature maps
		 short unsigned int IFMDim,               	// Length of input sequence
		 short unsigned int SIMDWidth,          	// Number of SIMD lanes
		 short unsigned int InputPrecision,         // Precisions for the input activation
		 short unsigned int InputIntPrecision,      // Input activation int bitwidth
		 short unsigned int OutputPrecision,		// Precisions for the output activation
		 short unsigned int OutputIntPrecision		// Output activation int bitwidth
>
void Relu1D(hls::stream<ap_uint<SIMDWidth * InputPrecision> > & in,
		    hls::stream<ap_uint<SIMDWidth * OutputPrecision> > & out);


template<
	short unsigned int KernelDim,        		// Conv kernel dim
	short unsigned int IFMChannels,				// Number of input feature maps
	short unsigned int IFMDim,               	// Length of input sequence
	short unsigned int Stride,					// Stride of kernel
	short unsigned int Padding,					// Padding alongside kernel
	short unsigned int OFMChannels,				// number of output feature maps
	short unsigned int OFMDim,               	// Length of output sequence // OFMDim = IFMDim/KernelDim
	short unsigned int SIMDWidth,          		// Number of SIMD lanes // NOT USED YET
	short unsigned int InputPrecision,        	// Precisions for the input activation
	short unsigned int InputIntPrecision,      	// Input activation int bitwidth
	short unsigned int OutputPrecision,        	// Precisions for the output activation
	short unsigned int OutputIntPrecision      	// Output activation int bitwidth
>
void MaxPool1D(hls::stream<ap_uint<SIMDWidth * InputPrecision> > & in,
			hls::stream<ap_uint<SIMDWidth * OutputPrecision> > & out);
