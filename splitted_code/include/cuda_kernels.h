// cuda_kernels.h
#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

void CHECK(const cudaError_t call);

void writeDeviceInfo(const char *filename, int device);

int setBestDevice();

void writeAllInfoToFile(const char *filename, int device);

__device__ double computeDistance(double *point1, double *point2, int metric, int exp, int num_features);

__global__ void knnDistances(double *trainData, double *testData, double *distances, int trainSize, int testSize, int metric, int exp, int num_features);

__device__ void swap(double *distances, int *indexes, int i, int j);

__device__ void bubbleSort(double *distances, int *indexes, int startIdx, int endIdx);

extern "C" __global__ void knnSortPredict(double *distances, int trainSize, int *indexes, int k, int *predictions, int *trainLabels, int sharedMemoryIdx, int alpha, int beta, int classes);



#endif // CUDA_KERNELS_H
