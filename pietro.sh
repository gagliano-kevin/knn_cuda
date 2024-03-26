#!/bin/bash

echo -e "\n************************************************ KNN ALGORITHM TESTS ************************************************\n"

echo -e "Checking for required tools and libraries...\n"


# Check for nvcc
if ! command -v nvcc &> /dev/null; then
    echo -e "nvcc could not be found\n"
    echo -e "Please make sure you have CUDA installed and nvcc is in your PATH\n"
    exit
fi
echo -e "--> nvcc found\n"

# Check for gcc
if ! command -v gcc &> /dev/null; then
    echo -e "gcc could not be found\n"
    echo -e "Please make sure you have gcc installed and it is in your PATH\n"
    exit
fi
echo -e "--> gcc found\n\n"


echo -e "Running tests on KNN algorithm...\n\n"

# --------------------- TEST ON ARTIFICIAL FEATURES --------------------- #

echo -e "-------------------------------------------- TEST ON ARTIFICIAL FEATURES --------------------------------------------\n"

echo -e "Compiling artificial_features.cu\n"

# Compile CUDA source file
nvcc source/tests/artificial_features.cu -o par_artificial_features

echo -e "Compiling artificial_features.c\n"

# Compile C source file
gcc source/tests/artificial_features.c -o seq_artificial_features -lm


echo -e "Running artificial_features.cu\n"
./par_artificial_features


echo -e "Running artificial_features.c\n"

# Run the compiled C code
./seq_artificial_features

# Clean up 
rm par_artificial_features seq_artificial_features

echo -e "Test on artificial data with growing features DONE\n\n\n"

# ----------------------------- TEST ON TRAINING SET SIZES ----------------------------- #

echo -e "-------------------------------------------- TEST ON TRAINING SET SIZES --------------------------------------------\n"

echo -e "Compiling artificial_trainSizes.cu\n"

# Compile CUDA source file
nvcc source/tests/artificial_trainSizes.cu -o par_artificial_trainSizes

echo -e "Compiling artificial_trainSizes.c\n"

# Compile C source file
gcc source/tests/artificial_trainSizes.c -o seq_artificial_trainSizes -lm


echo -e "Running artificial_trainSizes.cu\n"
./par_artificial_trainSizes

echo -e "Running artificial_trainSizes.c\n"

# Run the compiled C code
./seq_artificial_trainSizes

# Clean up 
rm par_artificial_trainSizes seq_artificial_trainSizes

echo -e "Test on artificial data with growing trainig set sizes DONE\n\n\n"



# ---------------------------------- TEST ON TEST SET SIZES ---------------------------------- #

echo -e "---------------------------------------------- TEST ON TEST SET SIZES ----------------------------------------------\n"

echo -e "Compiling artificial_testSizes.cu\n"

# Compile CUDA source file
nvcc source/tests/artificial_testSizes.cu -o par_artificial_testSizes

echo -e "Compiling artificial_testSizes.c\n"

# Compile C source file
gcc source/tests/artificial_testSizes.c -o seq_artificial_testSizes -lm


echo -e "Running artificial_testSizes.cu\n"
./par_artificial_testSizes


echo -e "Running artificial_testSizes.c\n"

# Run the compiled C code
./seq_artificial_testSizes

# Clean up 
rm par_artificial_testSizes seq_artificial_testSizes

echo -e "Test on artificial data with growing test set sizes DONE\n\n\n"


# ---------------------------------- TEST ON K PARAMETER ---------------------------------- #

echo -e "------------------------------------------------ TEST ON K PARAMETER ------------------------------------------------\n"

echo -e "Compiling artificial_k.cu\n"

# Compile CUDA source file
nvcc source/tests/artificial_k.cu -o par_artificial_k

echo -e "Compiling artificial_k.c\n"

# Compile C source file
gcc source/tests/artificial_k.c -o seq_artificial_k -lm


echo -e "Running artificial_k.cu\n"
./par_artificial_k 


echo -e "Running artificial_k.c\n"

# Run the compiled C code
./seq_artificial_k

# Clean up 
rm par_artificial_k seq_artificial_k

echo -e "Test on artificial data with growing k parameter DONE\n\n\n"


# --------------------------------------- TEST ON ALPHA PARAMETER --------------------------------------- #

echo -e "---------------------------------------------- TEST ON ALPHA PARAMETER ----------------------------------------------\n"

echo -e "Compiling artificial_alpha.cu\n"

# Compile CUDA source file
nvcc source/tests/artificial_alpha.cu -o par_artificial_alpha


echo -e "Running artificial_alpha.cu\n"
./par_artificial_alpha 


# Clean up 
rm par_artificial_alpha 

echo -e "Test on artificial data with growing alpha parameter DONE\n\n\n"


# ----------------------------- TEST ON BLOCKDIM KNN_DISTANCES PARAMETERS ----------------------------- #

echo -e "------------------------------------- TEST ON BLOCKDIM KNN_DISTANCES PARAMETERS -------------------------------------\n"

echo -e "Compiling artificial_blockDims.cu\n"

# Compile CUDA source file
nvcc source/tests/artificial_blockDims.cu -o par_artificial_blockDims


echo -e "Running artificial_blockDims.cu\n"
./par_artificial_blockDims 


# Clean up 
rm par_artificial_blockDims

echo -e "Test on artificial data with growing block dimensions DONE\n\n\n"


# ---------------------------------- TEST ON IRIS DATASET ---------------------------------- #

echo -e "------------------------------------------------ TEST ON IRIS DATASET -----------------------------------------------\n"

cd source

echo -e "Compiling par_knn_iris.cu\n"

# Compile CUDA source file
nvcc par_knn_iris.cu -o par_knn_iris

echo -e "Compiling seq_knn_iris.c\n"

# Compile C source file
gcc seq_knn_iris.c -o seq_knn_iris -lm


echo -e "Running par_knn_iris.cu\n"
./par_knn_iris 


echo -e "Running seq_knn_iris.c\n"

# Run the compiled C code
./seq_knn_iris

# Clean up 
rm par_knn_iris seq_knn_iris

echo -e "Test on iris dataset DONE\n\n\n"

mv iris ../iris

cd ..


# ---------------------------------- TEST ON DIABETES DATASET ---------------------------------- #

echo -e "---------------------------------------------- TEST ON DIABETES DATASET ---------------------------------------------\n"

cd source

echo -e "Compiling par_knn_diabetes.cu\n"

# Compile CUDA source file
nvcc par_knn_diabetes.cu -o par_knn_diabetes

echo -e "Compiling seq_knn_diabetes.c\n"

# Compile C source file
gcc seq_knn_diabetes.c -o seq_knn_diabetes -lm

# Run the compiled CUDA code with nvprof if available  and redirect output to txt file, otherwise directly execute it

echo -e "Running par_knn_diabetes.cu\n"
./par_knn_diabetes 


echo -e "Running seq_knn_diabetes.c\n"

# Run the compiled C code
./seq_knn_diabetes

# Clean up 
rm par_knn_diabetes seq_knn_diabetes

echo -e "Test on diabetes dataset DONE\n\n\n"

mv diabetes ../diabetes

cd ..


# Get the username of the currently logged-in user
username=$(whoami)

# Concatenate the username with the suffix '_results'
dir_name="${username}_results"

echo -e "Creating directory ./$dir_name to store all the results\n\n"

# Check if the directory already exists
if [ ! -d "./$dir_name" ]; then
    # Create a directory with the concatenated name
    mkdir -p "./$dir_name"
    echo -e "Directory './$dir_name' created successfully\n\n"
else
    echo -e "Directory './$dir_name' already exists\n\n"
    # In case the directory already exists, add the current date and time to the directory name
    echo -e "Adding current date and time to the directory name\n\n"
    dir_name="${username}_results_$(date +'%Y-%m-%d_%H-%M-%S')"
    mkdir -p "./$dir_name"
    echo -e "Directory './$dir_name' created successfully\n\n"
fi


# Move direcotires in result directory
mv ./sw_hw_info "./$dir_name/"
mv ./iris "./$dir_name/"
mv ./diabetes "./$dir_name/"
mv ./artificial_features "./$dir_name/"
mv ./artificial_trainSizes "./$dir_name/"
mv ./artificial_testSizes "./$dir_name/"
mv ./artificial_k "./$dir_name/"
mv ./artificial_alpha "./$dir_name/"
mv ./artificial_blockDims "./$dir_name/"

echo -e "All tests done. Results stored in directory ./$dir_name\n"
