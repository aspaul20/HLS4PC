# HLS4PC: A Parametrizable Framework For Accelerating Point-Based 3D Point Cloud Models on FPGA

## The code for the HLS4PC framework will be shared soon.
# HLS4PC: A Parametrizable Framework For Accelerating Point-Based 3D Point Cloud Models on FPGA
## HLS4PC

HLS4PC is a parametrizable framework for accelerating point-based 3D point cloud models on FPGA. This repository contains source code, synthesis scripts, and data files for the publication "HLS4PC: A Parametrizable Framework For Accelerating Point-Based 3D Point Cloud Models on FPGA".


## Directory Structure

- **src/**: HLS source files implementing core algorithms (e.g., Farthest Point Sampling, k-NN, neural network layers).
- **synth/**: Top-level synthesis files for Vivado HLS or similar tools.
- **data/**: Example input/output data and model weights for testing and benchmarking.
- **tb/**: Testbench files for simulation and verification.

## Getting Started

1. **Requirements**
   - Vivado HLS (2018.3 or later recommended)
   - C++17 compatible compiler (for simulation)
   - Python 3.x (optional, for data preparation)

2. **Build and Synthesize**
   - Navigate to the `synth/` directory.
   - Open your preferred HLS tool and import the desired top-level file (e.g., `fps_top.cpp`).
   - Run C simulation, synthesis, and implementation as needed.

3. **Testing**
   - Use the data in `data/` for simulation inputs.
   - Testbenches in `tb/` can be used for functional verification.

## Key Components

- **k-Nearest Neighbors (k-NN):** [`src/nn.hpp`](src/nn.hpp)
- **Uniform Random Sampling (URS):** [`src/urs.hpp`](src/urs.hpp)
- **Farthest Point Sampling (FPS):** [`src/fps.hpp`](src/fps.hpp)
- **Local Grouper Module with FPS:** [`src/fps.hpp`](src/fps.hpp)
- **Local Grouper Module with URS:** [`src/urs.hpp`](src/urs.hpp)
- **Types and Utilities:** [`src/types.hpp`](src/types.hpp)
- **HLS4PC Main Header:** [`src/hls4pc.hpp`](src/hls4pc.hpp)

## Example Data Files

- `input_points_S1.txt`: Input point cloud data.
- `input_embeddings_S1.txt`: Input feature embeddings.
- `q_81_weights.txt`, `q_81_biases.txt`: Example neural network weights and biases.

## Citation

If you use this framework in your research, please cite the corresponding paper.

## License

TBD