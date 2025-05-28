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
                      hls::stream<ap_uint<SIMDWidth * Precision> >& out)
{
    const unsigned int synapseFold = IFMChannels / SIMDWidth;
    ap_uint<SIMDWidth * Precision> inputBuf[KernelDim][synapseFold];
#pragma HLS ARRAY_PARTITION variable=inputBuf complete dim=2

    for (unsigned int r = 0; r < KernelDim; r++) {
        for (unsigned int c = 0; c < synapseFold; c++) {
            #pragma HLS PIPELINE II=1
            inputBuf[r][c] = in.read();
        }
    }

    ap_uint<2> current = 0;

    for (unsigned int ofm_iter = 0; ofm_iter < OFMDim; ofm_iter++) {
        for (unsigned int nm = 0; nm < (OFMChannels / PECount); nm++) {
            for (unsigned int c = 0; c < synapseFold; c++) {
                for (unsigned int k = 0; k < KernelDim; k++) {
                    #pragma HLS PIPELINE II=1
                    out.write( inputBuf[(current + k) % KernelDim][c] );
                }
            }
        }

        if (ofm_iter < IFMDim - KernelDim) {
            for (unsigned int c = 0; c < synapseFold; c++) {
                #pragma HLS PIPELINE II=1
                inputBuf[current][c] = in.read();
            }
        }
        current = (current + 1) % KernelDim;
    }
}


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
			   const ap_uint<BiasPrecision> biasMem [PECount][OFMChannels / PECount])
{

	typedef ap_fixed<InputPrecision, InputIntPrecision, AP_RND_ZERO, AP_WRAP> Input_t;
	typedef ap_fixed<BiasPrecision, BiasIntPrecision, AP_RND_ZERO, AP_WRAP> Bias_t;
	typedef ap_fixed<WeightsPrecision, WeightsIntPrecision, AP_RND_ZERO, AP_WRAP> Weights_t;
	typedef ap_fixed<MulPrecision, MulIntPrecision, AP_RND_ZERO, AP_WRAP> Mul_t;
	typedef ap_fixed<AccPrecision, AccIntPrecision, AP_RND_ZERO, AP_WRAP> Acc_t;
	typedef ap_fixed<OutputPrecision, OutputIntPrecision, AP_RND_ZERO, AP_WRAP> Output_t;

    const unsigned int neuronFold = OFMChannels / PECount;
    const unsigned int synapseFold = KernelDim * IFMChannels / SIMDWidth;

    Acc_t macRegisters[PECount];
#pragma HLS ARRAY_PARTITION variable=macRegisters complete dim=1

    loop_init:for(unsigned int pe = 0; pe < PECount; pe++)
    {
	#pragma HLS UNROLL

        macRegisters[pe] = 0;
    }

    loop_ofmChannels:for (unsigned int ofm_iter = 0; ofm_iter < OFMDim; ofm_iter++){
        loop_neuronFold:for (unsigned int nm = 0; nm < neuronFold; nm++){
            loop_synapseFold:for (unsigned int sf = 0; sf < synapseFold; sf++){
			#pragma HLS PIPELINE II=1

				 ap_uint<InputPrecision * SIMDWidth> input = in.read();

				loop_pe:for (unsigned int pe = 0; pe < PECount; pe++)
				{
				#pragma HLS UNROLL

					Acc_t tmpMac = macRegisters[pe];

					loop_simd:for(unsigned int simd = 0; simd < SIMDWidth; simd++)
					{
					#pragma HLS UNROLL

						Mul_t mul;

						unsigned int lowBit = simd * InputPrecision;
						unsigned int highBit = (simd + 1) * InputPrecision - 1;
						ap_int<InputPrecision> temp_input = input(highBit, lowBit);
						Input_t data = *reinterpret_cast<Input_t *>(&temp_input);
						ap_int<WeightsPrecision> temp_weight = weightMem[pe][simd][nm * synapseFold + sf];
						Weights_t weight = *reinterpret_cast<Weights_t *>(&temp_weight);
						mul = data * weight;
						tmpMac += mul;
					}

					macRegisters[pe] = tmpMac;
				}

				if(sf == synapseFold - 1)
				{
					ap_uint<PECount * OutputPrecision> output;

					for (unsigned int pe = 0; pe < PECount; pe++)
					{
					#pragma HLS UNROLL

						Output_t result;

						ap_int<BiasPrecision> temp = biasMem[pe][nm];
						Bias_t bias = *reinterpret_cast<Bias_t *>(&temp);
						macRegisters[pe] = macRegisters[pe] + (Acc_t)bias;

						result = (Output_t)macRegisters[pe];
						unsigned int lowBit = pe * OutputPrecision;
						unsigned int highBit = (pe + 1) * OutputPrecision - 1;
						ap_uint<OutputPrecision> output_temp = *reinterpret_cast<ap_uint<OutputPrecision> *>(&result);
						output(highBit, lowBit) = output_temp;

						macRegisters[pe] = 0;

					}

					out.write(output);
				}
			}
		}
    }

}


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
				  hls::stream<ap_uint<SIMDWidth * Precision> > & out)
{

	const unsigned int neuronFold = OFMChannels / PECount;
	const unsigned int synapseFold = IFMChannels / SIMDWidth;
    const unsigned int read_indices[KernelDim][KernelDim] = {{0}};

	ap_uint<SIMDWidth * Precision> inputBuf[KernelDim][synapseFold];

	for(unsigned int ptr_k = 0; ptr_k < KernelDim; ptr_k++){
		for(unsigned int ptr_simd = 0; ptr_simd < synapseFold; ptr_simd++){
			#pragma HLS PIPELINE II=1
			inputBuf[ptr_k][ptr_simd] = in.read();
		}
	}

	ap_uint<2> read_index = 0;
	for(unsigned int ofm_iter = 0; ofm_iter < OFMDim; ofm_iter++, read_index++){
		for(unsigned int nm = 0; nm < neuronFold; nm++){
			for(unsigned int ptr_simd = 0; ptr_simd < synapseFold; ptr_simd++){
				for(unsigned int read_index_k = 0; read_index_k < KernelDim; read_index_k++){
					#pragma HLS PIPELINE II=1
					if(read_index == KernelDim){
						read_index = 0;
					}
					unsigned int ptr_k = read_indices[read_index][read_index_k];
					out.write(inputBuf[ptr_k][ptr_simd]);

					if(ofm_iter < IFMDim - KernelDim && read_index_k == 0 && nm == neuronFold - 1)
					{
						inputBuf[ptr_k][ptr_simd] = in.read();
					}
				}
			}
		}
	}
}


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
			   const ap_uint<BiasPrecision> biasMem [PECount][OFMChannels / PECount])
{

	typedef ap_fixed<InputPrecision, InputIntPrecision, AP_RND_ZERO, AP_WRAP> Input_t;
	typedef ap_fixed<BiasPrecision, BiasIntPrecision, AP_RND_ZERO, AP_WRAP> Bias_t;
	typedef ap_fixed<WeightsPrecision, WeightsIntPrecision, AP_RND_ZERO, AP_WRAP> Weights_t;
	typedef ap_fixed<MulPrecision, MulIntPrecision, AP_RND_ZERO, AP_WRAP> Mul_t;
	typedef ap_fixed<AccPrecision, AccIntPrecision, AP_RND_ZERO, AP_WRAP> Acc_t;
	typedef ap_fixed<OutputPrecision, OutputIntPrecision, AP_RND_ZERO, AP_WRAP> Output_t;

    const unsigned int neuronFold = OFMChannels / PECount;
    const unsigned int synapseFold = KernelDim * IFMChannels / SIMDWidth;

    Acc_t macRegisters[PECount];
#pragma HLS ARRAY_PARTITION variable=macRegisters complete dim=1

    loop_init:for(unsigned int pe = 0; pe < PECount; pe++)
    {
	#pragma HLS UNROLL

        macRegisters[pe] = 0;
    }

    loop_ofmChannels:for (unsigned int ofm_iter = 0; ofm_iter < OFMDim; ofm_iter++){
        loop_neuronFold:for (unsigned int nm = 0; nm < neuronFold; nm++){
            loop_synapseFold:for (unsigned int sf = 0; sf < synapseFold; sf++){
			#pragma HLS PIPELINE II=1

				 ap_uint<InputPrecision * SIMDWidth> input = in.read();

				loop_pe:for (unsigned int pe = 0; pe < PECount; pe++)
				{
				#pragma HLS UNROLL

					Acc_t tmpMac = macRegisters[pe];

					loop_simd:for(unsigned int simd = 0; simd < SIMDWidth; simd++)
					{
					#pragma HLS UNROLL

						Mul_t mul;

						unsigned int lowBit = simd * InputPrecision;
						unsigned int highBit = (simd + 1) * InputPrecision - 1;
						ap_int<InputPrecision> temp_input = input(highBit, lowBit);
						Input_t data = *reinterpret_cast<Input_t *>(&temp_input);


						ap_int<WeightsPrecision> temp_weight = weightMem[pe][simd][nm * synapseFold + sf];
						Weights_t weight = *reinterpret_cast<Weights_t *>(&temp_weight);

						mul = data * weight;
						tmpMac += mul;


					}

					macRegisters[pe] = tmpMac;
				}

				if(sf == synapseFold - 1)
				{
					ap_uint<PECount * OutputPrecision> output;

					for (unsigned int pe = 0; pe < PECount; pe++)
					{
					#pragma HLS UNROLL

						Output_t result;

						ap_int<BiasPrecision> temp = biasMem[pe][nm];
						Bias_t bias = *reinterpret_cast<Bias_t *>(&temp);
						macRegisters[pe] = macRegisters[pe] + (Acc_t)bias;

						result = (Output_t)macRegisters[pe];


						unsigned int lowBit = pe * OutputPrecision;
						unsigned int highBit = (pe + 1) * OutputPrecision - 1;
						ap_uint<OutputPrecision> output_temp = *reinterpret_cast<ap_uint<OutputPrecision> *>(&result);
						output(highBit, lowBit) = output_temp;

						macRegisters[pe] = 0;

					}

					out.write(output);
				}
			}
		}
    }

}

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
		   const ap_uint<BiasPrecision> biasMem [PECount][Neurons / PECount])
{

	typedef ap_fixed<InputPrecision, InputIntPrecision, AP_RND_ZERO, AP_WRAP> Input_t;
	typedef ap_fixed<BiasPrecision, BiasIntPrecision, AP_RND_ZERO, AP_WRAP> Bias_t;
	typedef ap_fixed<WeightsPrecision, WeightsIntPrecision, AP_RND_ZERO, AP_WRAP> Weights_t;
	typedef ap_fixed<MulPrecision, MulIntPrecision, AP_RND_ZERO, AP_WRAP> Mul_t;
	typedef ap_fixed<AccPrecision, AccIntPrecision, AP_RND_ZERO, AP_WRAP> Acc_t;
	typedef ap_fixed<OutputPrecision, OutputIntPrecision, AP_RND_ZERO, AP_WRAP> Output_t;

    const unsigned int neuronFold = Neurons / PECount;
    const unsigned int synapseFold = Inputs / SIMDWidth;

    ap_uint<SIMDWidth * InputPrecision> input;
    ap_uint<PECount * OutputPrecision> output;

	Acc_t macRegisters[PECount][neuronFold];
#pragma HLS ARRAY_PARTITION variable=macRegisters complete dim=1
#pragma HLS ARRAY_PARTITION variable=weightMem complete dim=2
#pragma HLS ARRAY_PARTITION variable=biasMem complete dim=2

	loop_init_ne:for(unsigned int ne = 0; ne < neuronFold; ne++)
	{

		loop_init_pe:for(unsigned int pe = 0; pe < PECount; pe++)
		{
		#pragma HLS UNROLL

			ap_int<BiasPrecision> temp_bias = biasMem[pe][ne];
			Bias_t bias = *reinterpret_cast<Bias_t *>(&temp_bias);

			macRegisters[pe][ne] = (Acc_t)bias;

		}
	}

	loop_dim:for(unsigned int sy = 0; sy < synapseFold; sy++)
	{

		input = in.read();

		loop_ne:for(unsigned int ne = 0; ne < neuronFold; ne++)
		{
			loop_pe:for(unsigned int pe = 0; pe < PECount; pe++)
			{
				Acc_t tmpMac = macRegisters[pe][ne];

				loop_simd:for (unsigned int simd = 0; simd < SIMDWidth; simd++)
				{
					Mul_t mul;

					unsigned int lowBit = simd * InputPrecision;
					unsigned int highBit = (simd + 1) * InputPrecision - 1;
					ap_int<InputPrecision> temp_data = input(highBit, lowBit);
					Input_t data = *reinterpret_cast<Input_t *>(&temp_data);

					ap_int<WeightsPrecision> temp_weight = weightMem[pe][simd][ne * synapseFold + sy];
					Weights_t weight = *reinterpret_cast<Weights_t *>(&temp_weight);

					mul = data * weight;
					tmpMac += mul;
				}

				macRegisters[pe][ne] = tmpMac;
			}
		}
	}

	loop_output_ne:for(unsigned int ne = 0; ne < neuronFold; ne++)
	{
	#pragma HLS PIPELINE

		loop_output_pe:for(unsigned int pe = 0; pe < PECount; pe++)
		{
		#pragma HLS UNROLL


			Output_t temp_reg = (Output_t)macRegisters[pe][ne];
			ap_uint<OutputPrecision> temp_output = *reinterpret_cast< ap_uint<OutputPrecision> *>(&temp_reg);

			unsigned int lowBit = pe * OutputPrecision;
			unsigned int highBit = (pe + 1) * OutputPrecision - 1;
			output(highBit, lowBit) = temp_output;
		}

		out.write(output);
	}

}

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
		    hls::stream<ap_uint<SIMDWidth * OutputPrecision> > & out)
{


	typedef ap_fixed<InputPrecision, InputIntPrecision, AP_RND_ZERO, AP_WRAP> Input_t;
	typedef ap_fixed<OutputPrecision, OutputIntPrecision, AP_RND_ZERO, AP_WRAP> Output_t;

	const unsigned int synapseFold = IFMChannels / SIMDWidth;
	const unsigned int duration = IFMDim * synapseFold;

	loop_dim:for (unsigned int i = 0; i < duration; i++)
	{
	#pragma HLS PIPELINE II=1

		ap_uint<SIMDWidth * InputPrecision> input = in.read();
		ap_uint<SIMDWidth * OutputPrecision> output;

		loop_simd:for (unsigned int simd = 0; simd < SIMDWidth; simd++)
		{
		#pragma HLS UNROLL

			unsigned int lowBiti = simd * InputPrecision;
			unsigned int highBiti = (simd + 1) * InputPrecision - 1;
			ap_int<InputPrecision> temp_data = input(highBiti, lowBiti);
			Input_t data = *reinterpret_cast<Input_t *>(&temp_data);

			Output_t result;

			if(data < (Input_t)0.0)
				result = (Output_t)0.0;
	    	else
				result = (Output_t)data;

			unsigned int lowBito = simd * OutputPrecision;
			unsigned int highBito = (simd + 1) * OutputPrecision - 1;
			ap_uint<OutputPrecision> output_temp = *reinterpret_cast<ap_uint<OutputPrecision> *>(&result);
			output(highBito, lowBito) = output_temp;
		}

		out.write(output);
	}
}

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
		           hls::stream<ap_uint<SIMDWidth * OutputPrecision> > & out)
{
	typedef ap_fixed<InputPrecision, InputIntPrecision, AP_RND_ZERO, AP_WRAP> Input_t;
	typedef ap_fixed<OutputPrecision, OutputIntPrecision, AP_RND_ZERO, AP_WRAP> Output_t;

	ap_uint<SIMDWidth * InputPrecision> buffer[IFMChannels / SIMDWidth];

	const unsigned int synapseFold = IFMChannels / SIMDWidth;
	unsigned int sf = 0;
	unsigned int init = 1;

	ap_uint<SIMDWidth * InputPrecision> output;

	loop_dim:for (unsigned int i = 0; i < synapseFold * IFMDim; i++)
	{
	#pragma HLS PIPELINE II=synapseFold

		if(init == 1)
		{
			buffer[sf] = in.read();
			sf++;

			if(sf == synapseFold)
			{
				sf = 0;
				init = 0;
			}
		}
		else
		{
			ap_int<SIMDWidth * InputPrecision> temp_data_0 = in.read();
			ap_int<SIMDWidth * InputPrecision> temp_data_1 = buffer[sf];

			loop_ch:for(unsigned int simd = 0; simd < SIMDWidth; simd++)
			{
			#pragma HLS UNROLL

				unsigned int lowBit = simd * InputPrecision;
				unsigned int highBit = (simd + 1) * InputPrecision - 1;

				ap_uint<InputPrecision> temp_data_0_in_ch = temp_data_0(highBit, lowBit);
				ap_uint<InputPrecision> temp_data_1_in_ch = temp_data_1(highBit, lowBit);

				Input_t data_0 = *reinterpret_cast<Input_t *>(&temp_data_0_in_ch);
				Input_t data_1 = *reinterpret_cast<Input_t *>(&temp_data_1_in_ch);

				Output_t result;

				if(data_0 > data_1)
				{
					result = data_0;
				}
				else
				{
					result = data_1;
				}

				ap_uint<OutputPrecision> output_temp = *reinterpret_cast< ap_uint<OutputPrecision> *>(&result);
				output(highBit, lowBit) = output_temp;

			}

			out.write(output);

			sf++;
			if(sf == synapseFold)
			{
				sf = 0;
				init = 1;
			}

		}
	}
}
