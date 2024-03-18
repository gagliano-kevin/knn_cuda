#!/bin/bash


# Create directory to store nvprof outputs
mkdir nvprof_outputs

# --------------------- TEST ON ARTIFICIAL FEATURES --------------------- #

# Compile CUDA source file
nvcc source/tests/artificial_features.cu -o par_artificial_features

# Compile C source file
gcc source/tests/artificial_features.c -o seq_artificial_features -lm

# Run the compiled CUDA code with nvprof and redirect output to txt file
nvprof ./par_artificial_features > >(tee nvprof_outputs/par_artificial_features_nvprof_output.txt) 2>&1

# Run the compiled C code
./seq_artificial_features

# Clean up 
rm par_artificial_features seq_artificial_features

# --------------------- TEST ON TRAIN SIZES --------------------- #