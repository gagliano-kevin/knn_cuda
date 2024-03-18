#!/bin/bash


# Create directory to store nvprof outputs
mkdir nvprof_outputs

# --------------------- TEST ON ARTIFICIAL FEATURES --------------------- #

echo -e "Compiling artificial_features.cu\n"

# Compile CUDA source file
nvcc source/tests/artificial_features.cu -o par_artificial_features

echo -e "Compiling artificial_features.c\n"

# Compile C source file
gcc source/tests/artificial_features.c -o seq_artificial_features -lm

echo -e "Running artificial_features.cu, nvprof output redirected to nvprof_outputs/par_artificial_features_nvprof_output.txt\n"

# Run the compiled CUDA code with nvprof and redirect output to txt file
nvprof ./par_artificial_features > >(tee nvprof_outputs/par_artificial_features_nvprof_output.txt) 2>&1

echo -e "Running artificial_features.c\n"

# Run the compiled C code
./seq_artificial_features

# Clean up 
rm par_artificial_features seq_artificial_features

echo -e "Test on artificial features done\n"

# --------------------- TEST ON TRAIN SIZES --------------------- #