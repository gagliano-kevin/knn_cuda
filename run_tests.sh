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
nvprof ./par_artificial_features > nvprof_outputs/par_artificial_features_nvprof_output.txt 2>&1

echo -e "Running artificial_features.c\n"

# Run the compiled C code
./seq_artificial_features

# Clean up 
rm par_artificial_features seq_artificial_features

echo -e "Test on artificial data with growing features done\n\n\n"

# ----------------------------- TEST ON TRAINING SET SIZES ----------------------------- #

echo -e "Compiling artificial_trainSizes.cu\n"

# Compile CUDA source file
nvcc source/tests/artificial_trainSizes.cu -o par_artificial_trainSizes

echo -e "Compiling artificial_trainSizes.c\n"

# Compile C source file
gcc source/tests/artificial_trainSizes.c -o seq_artificial_trainSizes -lm

echo -e "Running artificial_trainSizes.cu, nvprof output redirected to nvprof_outputs/par_artificial_trainSizes_nvprof_output.txt\n"

# Run the compiled CUDA code with nvprof and redirect output to txt file
nvprof ./par_artificial_trainSizes > nvprof_outputs/par_artificial_trainSizes_nvprof_output.txt 2>&1

echo -e "Running artificial_trainSizes.c\n"

# Run the compiled C code
./seq_artificial_trainSizes

# Clean up 
rm par_artificial_trainSizes seq_artificial_trainSizes

echo -e "Test on artificial data with growing trainig set sizes done\n\n\n"


# ----------------------------- TEST ON TEST SET SIZES ----------------------------- #

echo -e "Compiling artificial_testSizes.cu\n"

# Compile CUDA source file
nvcc source/tests/artificial_testSizes.cu -o par_artificial_testSizes

echo -e "Compiling artificial_testSizes.c\n"

# Compile C source file
gcc source/tests/artificial_testSizes.c -o seq_artificial_testSizes -lm

echo -e "Running artificial_testSizes.cu, nvprof output redirected to nvprof_outputs/par_artificial_testSizes_nvprof_output.txt\n"

# Run the compiled CUDA code with nvprof and redirect output to txt file
nvprof ./par_artificial_testSizes > nvprof_outputs/par_artificial_testSizes_nvprof_output.txt 2>&1

echo -e "Running artificial_testSizes.c\n"

# Run the compiled C code
./seq_artificial_testSizes

# Clean up 
rm par_artificial_testSizes seq_artificial_testSizes

echo -e "Test on artificial data with growing test set sizes done\n\n\n"

# ----------------------------- TEST ON K PARAMETER ----------------------------- #

echo -e "Compiling artificial_k.cu\n"

# Compile CUDA source file
nvcc source/tests/artificial_k.cu -o par_artificial_k

echo -e "Compiling artificial_k.c\n"

# Compile C source file
gcc source/tests/artificial_k.c -o seq_artificial_k -lm

echo -e "Running artificial_k.cu, nvprof output redirected to nvprof_outputs/par_artificial_k_nvprof_output.txt\n"

# Run the compiled CUDA code with nvprof and redirect output to txt file
nvprof ./par_artificial_k > nvprof_outputs/par_artificial_k_nvprof_output.txt 2>&1

echo -e "Running artificial_k.c\n"

# Run the compiled C code
./seq_artificial_k

# Clean up 
rm par_artificial_k seq_artificial_k

echo -e "Test on artificial data with growing k parameter done\n\n\n"

# ----------------------------- TEST ON ALPHA PARAMETER ----------------------------- #

echo -e "Compiling artificial_alpha.cu\n"

# Compile CUDA source file
nvcc source/tests/artificial_alpha.cu -o par_artificial_alpha

echo -e "Running artificial_k.cu, nvprof output redirected to nvprof_outputs/par_artificial_alpha_nvprof_output.txt\n"

# Run the compiled CUDA code with nvprof and redirect output to txt file
nvprof ./par_artificial_alpha > nvprof_outputs/par_artificial_alpha_nvprof_output.txt 2>&1

# Clean up 
rm par_artificial_alpha 

echo -e "Test on artificial data with growing alpha parameter done\n\n\n"

# ----------------------------- TEST ON BLOCKDIM KNN_DISTANCES PARAMETERS ----------------------------- #

echo -e "Compiling artificial_blockDims.cu\n"

# Compile CUDA source file
nvcc source/tests/artificial_blockDims.cu -o par_artificial_blockDims

echo -e "Running artificial_blockDims.cu, nvprof output redirected to nvprof_outputs/par_artificial_blockDims_nvprof_output.txt\n"

# Run the compiled CUDA code with nvprof and redirect output to txt file
nvprof ./par_artificial_blockDims > nvprof_outputs/par_artificial_blockDims_nvprof_output.txt 2>&1

# Clean up 
rm par_artificial_blockDims

echo -e "Test on artificial data with growing block dimensions done\n\n\n"

# ----------------------------- TEST ON IRIS DATASET ----------------------------- #

cd source

echo -e "Compiling par_knn_iris.cu\n"

# Compile CUDA source file
nvcc par_knn_iris.cu -o par_knn_iris

echo -e "Compiling seq_knn_iris.c\n"

# Compile C source file
gcc seq_knn_iris.c -o seq_knn_iris -lm

echo -e "Running par_knn_iris.cu, nvprof output redirected to nvprof_outputs/par_knn_iris_nvprof_output.txt\n"

# Run the compiled CUDA code with nvprof and redirect output to txt file
nvprof ./par_knn_iris > ../nvprof_outputs/par_knn_iris_nvprof_output.txt 2>&1

echo -e "Running seq_knn_iris.c\n"

# Run the compiled C code
./seq_knn_iris

# Clean up 
rm par_knn_iris seq_knn_iris

echo -e "Test on iris dataset done\n\n\n"

mv iris ../iris

cd ..

# ----------------------------- TEST ON DIABETES DATASET ----------------------------- #

cd source

echo -e "Compiling par_knn_diabetes.cu\n"

# Compile CUDA source file
nvcc par_knn_diabetes.cu -o par_knn_diabetes

echo -e "Compiling seq_knn_diabetes.c\n"

# Compile C source file
gcc seq_knn_diabetes.c -o seq_knn_diabetes -lm

echo -e "Running par_knn_diabetes.cu, nvprof output redirected to nvprof_outputs/par_knn_diabetes_nvprof_output.txt\n"

# Run the compiled CUDA code with nvprof and redirect output to txt file
nvprof ./par_knn_diabetes > ../nvprof_outputs/par_knn_diabetes_nvprof_output.txt 2>&1

echo -e "Running seq_knn_diabetes.c\n"

# Run the compiled C code
./seq_knn_diabetes

# Clean up 
rm par_knn_diabetes seq_knn_diabetes

echo -e "Test on diabetes dataset done\n\n\n"

mv diabetes ../diabetes

cd ..


# Get the username of the currently logged-in user
username=$(whoami)

# Concatenate the username with the suffix '_results'
dir_name="${username}_results"

# Check if the directory already exists
if [ ! -d "./$dir_name" ]; then
    # Create a directory with the concatenated name
    mkdir -p "./$dir_name"
    echo "Directory './$dir_name' created successfully."
else
    echo "Directory './$dir_name' already exists."
fi

# Move direcotires in result directory
mv ./nvprof_outputs "./$dir_name/"
mv ./sw_hw_info "./$dir_name/"
mv ./iris "./$dir_name/"
mv ./diabetes "./$dir_name/"
mv ./artificial_features "./$dir_name/"
mv ./artificial_trainSizes "./$dir_name/"
mv ./artificial_testSizes "./$dir_name/"
mv ./artificial_k "./$dir_name/"
mv ./artificial_alpha "./$dir_name/"
mv ./artificial_blockDims "./$dir_name/"
