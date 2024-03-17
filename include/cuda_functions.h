#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

#include "../include/common.h"

#include <cuda_runtime.h>
#include <sys/sysinfo.h>



#define FILE_NAME __FILE__
#define LINE __LINE__




void CHECK(const cudaError_t call) {
    const cudaError_t error = call;
    if(error != cudaSuccess){
        printf("Error: %s:%d, ", FILE_NAME, LINE);
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));
        exit(1);
    }
}


// Compute distance between two points based on the selected metric
__device__ double computeDistance(double *point1, double *point2, int metric, int exp, int num_features) {
    double distance = 0.0;
    if (metric == 1) { // Euclidean distance
        for (int i = 0; i < num_features; i++) {
            distance += (point1[i] - point2[i]) * (point1[i] - point2[i]);
        }
        distance = sqrt(distance);
    } else if (metric == 2) { // Manhattan distance
        for (int i = 0; i < num_features; i++) {
            distance += fabs(point1[i] - point2[i]);
        }
    } else if (metric == 3) { // Minkowski distance with p = exp
        double sum = 0.0;
        for (int i = 0; i < num_features; i++) {
            sum += pow(fabs(point1[i] - point2[i]), exp);
        }
        distance = pow(sum, 1.0 / (float)exp);
    }
    return distance;
}


// distances matrix has size testSize x TrainSize 
__global__ void knnDistances(double *trainData, double *testData, double *distances, int trainSize, int testSize, int metric, int exp, int num_features) { 
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * trainSize + ix; 
    if(ix < trainSize && iy < testSize){
        distances[idx]=computeDistance(&trainData[ix * num_features], &testData[iy * num_features], metric, exp, num_features);
    }
}


__device__ void swap(double *distances, int *indexes, int i, int j) {
    double tempDist = distances[i];
    int tempInd = indexes[i];
    distances[i] = distances[j];
    indexes[i] = indexes[j];
    distances[j] = tempDist;
    indexes[j] = tempInd;
}


__device__ void bubbleSort(double *distances, int *indexes, int startIdx, int endIdx) {
    for (int i = startIdx; i < endIdx - 1; i++) {
        for (int j = startIdx; j < endIdx - i + startIdx - 1; j++) {
            if (distances[j] > distances[j + 1]) {
                swap(distances, indexes, j, j + 1);
            }
        }
    }
}


// exter "C" is used to avoid name mangling
extern "C" __global__ void knn(double *distances, int trainSize, int *indexes, int k, int *predictions, int *trainLabels, int sharedMemoryIdx, int alpha, int beta, int classes) {
    int row = blockIdx.x;
    int portion = (int)trainSize / blockDim.x;

    int startIdx = row * trainSize + threadIdx.x * portion;
    int endIdx = startIdx + portion;
    // last thread block takes care of the remaining portion of the dataset
    if(threadIdx.x == blockDim.x - 1){
        endIdx = startIdx + (trainSize - (blockDim.x - 1) * portion);
    }

    bubbleSort(distances, indexes, startIdx, endIdx);
    
    // Dynamically allocate shared memory for distances and indexes
    extern __shared__ double sharedMemory[];
    double *sharedDistances = sharedMemory;
    int *sharedIndexes = (int*)&sharedMemory[sharedMemoryIdx];

    double *sharedDistances2 = &sharedDistances[k * blockDim.x];
    int *sharedIndexes2 = &sharedIndexes[k * blockDim.x];

    for (int i = 0; i < k; i++) {
        sharedDistances[threadIdx.x * k + i] = distances[startIdx + i];
        sharedIndexes[threadIdx.x * k + i] = indexes[startIdx + i];
    }
    
    __syncthreads();


    int iter = 1;
    int workers = blockDim.x;
    trainSize = k * workers;

    while(trainSize >= beta * k){
        iter++;
        workers = (int)workers / alpha;
        portion = alpha * k;
        if(iter % 2 == 0){      // even iteration
            if(threadIdx.x < workers){
                int startIdx = threadIdx.x * portion;
                int endIdx = startIdx + portion;
                if(threadIdx.x == workers - 1){ // last thread 
                    endIdx = trainSize;
                }
                bubbleSort(sharedDistances, sharedIndexes, startIdx, endIdx);
                for (int i = 0; i < k; i++) {
                    sharedDistances2[threadIdx.x * k + i] = sharedDistances[startIdx + i];
                    sharedIndexes2[threadIdx.x * k + i] = sharedIndexes[startIdx + i];
                }
                
                __syncthreads();
            }
        } else {        // odd iteration
            if(threadIdx.x < workers){
                int startIdx = threadIdx.x * portion;
                int endIdx = startIdx + portion;
                if(threadIdx.x == workers - 1){ // last thread 
                    endIdx = trainSize;
                }
                bubbleSort(sharedDistances2, sharedIndexes2, startIdx, endIdx);
                for (int i = 0; i < k; i++) {
                    sharedDistances[threadIdx.x * k + i] = sharedDistances2[startIdx + i];
                    sharedIndexes[threadIdx.x * k + i] = sharedIndexes2[startIdx + i];
                }
                
                __syncthreads();
            }
        }
        trainSize = k * workers;
    }

    // last iteration (sequential with 1 worker)
    if(threadIdx.x == 0){
        double *distances;
        int *indexes;
        if(iter % 2 == 0){
            distances = sharedDistances2;
            indexes = sharedIndexes2;
        } else {
            distances = sharedDistances;
            indexes = sharedIndexes;
        }
        bubbleSort(distances, indexes, 0, trainSize);

        // Nearest class election
        int* classCounts = new int[classes];
        // Initialize all elements to zero
        for (int i = 0; i < classes; ++i) {
            classCounts[i] = 0;
        }
        for (int i = 0; i < k; i++){        
            classCounts[trainLabels[indexes[i]]]++;
        }

        int max = 0; 
        int maxClass = -1;
        for (int i = 0; i < classes; i++){        
            if(classCounts[i] > max){
                max = classCounts[i];
                maxClass = i;
            }
        }
        predictions[row] = maxClass;
        delete[] classCounts;
    }
}

void appendResultsToFile(int errorCount, int testSize, const char *filename, const char *dirname, int trainSize, int features, int k, int metric, int exp, unsigned int *distDim, unsigned int *predDim, int workers, int alpha, int beta, double kernelTime1, double kernelTime2, int sharedMemory, int maxSharedMemory, int sharedWorkers){
    createDirectory(dirname); 

    char path[256]; // Assuming max path length of 256 characters
    snprintf(path, sizeof(path), "%s%s", dirname, filename);
    
    FILE *file = fopen(path, "a");
    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }

    fprintf(file, "Kernel launch information:\n");
    fprintf(file, "Grid dimension in knnDistances kernel: %u , %u\n", distDim[0], distDim[1]);
    fprintf(file, "Block dimension in knnDistances kernel: %u , %u\n", distDim[2], distDim[3]);
    fprintf(file, "Grid dimension in knn kernel: %u , %u\n",predDim[0], predDim[1]);
    fprintf(file, "Block dimension in knn kernel: %u , %u\n", predDim[2], predDim[3]);
    fprintf(file, "Number of workers: %d\n", workers);
    fprintf(file, "Shared workers (threads that initially operate in shared memory partitions): %d\n", sharedWorkers);
    fprintf(file, "Factor alpha: %d\n", alpha);
    fprintf(file, "Factor beta: %d\n", beta);
    fprintf(file, "Shared memory used: %d bytes \t<---->\t Max shared memory per block: %d bytes\n", sharedMemory, maxSharedMemory);
    fprintf(file, "knnDistances execution time %f sec\n", kernelTime1);
    fprintf(file, "knn execution time %f sec\n", kernelTime2);

    fprintf(file, "\nData information:\n");
    fprintf(file, "Training data size: %d\n", trainSize);
    fprintf(file, "Test data size: %d\n", testSize);
    fprintf(file, "Number of features: %d\n", features);

    fprintf(file, "\nKNN Parameters:\n");
    fprintf(file, "k: %d\n", k);
    fprintf(file, "Distance Metric: ");
    if (metric == 1) {
        fprintf(file, "Euclidean\n");
    } else if (metric == 2) {
        fprintf(file, "Manhattan\n");
    } else if (metric == 3) {
        fprintf(file, "Minkowski (p=%d)\n", exp);
    }

    fprintf(file, "\nNumber of prediction errors: %d\n", errorCount);
    fprintf(file, "\n-------------------------------------------------------------------------------------------------------------------------------------------\n\n");


    fclose(file);
}


void writeResultsToFile(int * trainLabels, int *results, int errorCount, int testSize, const char *filename, const char *dirname, int trainSize, int features, int k, int metric, int exp, unsigned int *distDim, unsigned int *predDim, int workers, int alpha, int beta, double kernelTime1, double kernelTime2, int sharedMemory, int maxSharedMemory, int sharedWorkers) {
    
    createDirectory(dirname); 

    char path[256]; // Assuming max path length of 256 characters
    snprintf(path, sizeof(path), "%s%s", dirname, filename);
    
    FILE *file = fopen(path, "w");
    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }

    fprintf(file, "Kernel launch information:\n");
    fprintf(file, "Grid dimension in knnDistances kernel: %u , %u\n", distDim[0], distDim[1]);
    fprintf(file, "Block dimension in knnDistances kernel: %u , %u\n", distDim[2], distDim[3]);
    fprintf(file, "Grid dimension in knn kernel: %u , %u\n",predDim[0], predDim[1]);
    fprintf(file, "Block dimension in knn kernel: %u , %u\n", predDim[2], predDim[3]);
    fprintf(file, "Number of workers: %d\n", workers);
    fprintf(file, "Shared workers (threads that initially operate in shared memory partitions): %d\n", sharedWorkers);
    fprintf(file, "Factor alpha: %d\n", alpha);
    fprintf(file, "Factor beta: %d\n", beta);
    fprintf(file, "Shared memory used: %d bytes \t<---->\t Max shared memory per block: %d bytes\n", sharedMemory, maxSharedMemory);
    fprintf(file, "knnDistances execution time %f sec\n", kernelTime1);
    fprintf(file, "knn execution time %f sec\n", kernelTime2);

    fprintf(file, "\nData information:\n");
    fprintf(file, "Training data size: %d\n", trainSize);
    fprintf(file, "Test data size: %d\n", testSize);
    fprintf(file, "Number of features: %d\n", features);

    fprintf(file, "\nKNN Parameters:\n");
    fprintf(file, "k: %d\n", k);
    fprintf(file, "Distance Metric: ");
    if (metric == 1) {
        fprintf(file, "Euclidean\n");
    } else if (metric == 2) {
        fprintf(file, "Manhattan\n");
    } else if (metric == 3) {
        fprintf(file, "Minkowski (p=%d)\n", exp);
    }

    fprintf(file, "\nNumber of prediction errors: %d\n", errorCount);
    fprintf(file, "\nResults:\n");
    char outcome[25];
    for (int i = 0; i < testSize; ++i) {
        if(results[i] == trainLabels[i]){
            strcpy(outcome, "correctly classified");
        } else {
            strcpy(outcome, "incorrectly classified");
        }
        fprintf(file, "Test example %3d  %-22s -> Predicted class: %1d , Expected class: %1d\n", i + 1, outcome, results[i], trainLabels[i]);
    }

    fclose(file);

    printf("Execution results has been written to %s\n\n", path);
}



void writeDeviceInfo(const char *filename, int device){
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }

    int dev, driverVersion = 0, runtimeVersion = 0;

    dev = device;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    fprintf(file, "Device %d: \"%s\"\n", dev, deviceProp.name);

    CHECK(cudaDriverGetVersion(&driverVersion));
    CHECK(cudaRuntimeGetVersion(&runtimeVersion));

    fprintf(file, "CUDA Driver Version / Runtime Version %d.%d / %d.%d\n",
    driverVersion/1000, (driverVersion%100)/10,
    runtimeVersion/1000, (runtimeVersion%100)/10);

    fprintf(file, "CUDA Capability Major/Minor version number: %d.%d\n",
    deviceProp.major, deviceProp.minor);

    fprintf(file, "Total amount of global memory: %.2f GBytes (%llu bytes)\n",
    (float)deviceProp.totalGlobalMem/(pow(1024.0,3)),
    (unsigned long long) deviceProp.totalGlobalMem);

    fprintf(file, "GPU Clock rate: %.0f MHz (%0.2f GHz)\n",
    deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

    fprintf(file, "Memory Clock rate: %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);

    fprintf(file, "Memory Bus Width: %d-bit\n", deviceProp.memoryBusWidth);

    if (deviceProp.l2CacheSize) {
        fprintf(file, "L2 Cache Size: %d bytes\n", deviceProp.l2CacheSize);
    }

    fprintf(file, "Max Texture Dimension Size (x,y,z) "
    " 1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n",
    deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
    deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0], 
    deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);

    fprintf(file, "Max Layered Texture Size (dim) x layers 1D=(%d) x %d, 2D=(%d,%d) x %d\n",
    deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1],
    deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
    deviceProp.maxTexture2DLayered[2]);

    fprintf(file, "Total amount of constant memory: %lu bytes\n", deviceProp.totalConstMem);
    
    fprintf(file, "Total amount of shared memory per block: %lu bytes\n", deviceProp.sharedMemPerBlock);
    
    fprintf(file, "Total number of registers available per block: %d\n",  deviceProp.regsPerBlock);
    
    fprintf(file, "Warp size: %d\n", deviceProp.warpSize);
    
    fprintf(file, "Maximum number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
    
    fprintf(file, "Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    
    fprintf(file, "Maximum sizes of each dimension of a block: %d x %d x %d\n",
    deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    
    fprintf(file, "Maximum sizes of each dimension of a grid: %d x %d x %d\n",
    deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    
    fprintf(file, "Maximum memory pitch: %lu bytes\n", deviceProp.memPitch);

    fclose(file);
}


int setBestDevice(){
    int numDevices = 0;
    CHECK(cudaGetDeviceCount(&numDevices));

    if(numDevices == 0){
        printf("There are no available device(s) that support CUDA\n");
        return -1;
    }

    if(numDevices > 1) {
        printf("Detected %d CUDA capable device(s)\n", numDevices);
        int maxMultiprocessors = 0, maxDevice = 0;
        for (int device=0; device<numDevices; device++) {
            cudaDeviceProp props;
            CHECK(cudaGetDeviceProperties(&props, device));
            if (maxMultiprocessors < props.multiProcessorCount) {
                maxMultiprocessors = props.multiProcessorCount;
                maxDevice = device;
            }
        }
        CHECK(cudaSetDevice(maxDevice));
        cudaDeviceProp best_prop;
        CHECK(cudaGetDeviceProperties(&best_prop, maxDevice));
        printf("Setting Device %d : \"%s\"\n\n", maxDevice, best_prop.name);
        return maxDevice;
    } else {
        printf("Detected only one CUDA Device ...\n");
        int dev = 0;
        CHECK(cudaSetDevice(dev));
        cudaDeviceProp deviceProp;
        CHECK(cudaGetDeviceProperties(&deviceProp, dev));
        printf("Setting Device %d: \"%s\"\n\n", dev, deviceProp.name);
        return dev;
    }
}


int getMaxThreadsPerBlock(int device){
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, device));
    return deviceProp.maxThreadsPerBlock;
}


int getSharedMemoryPerBlock(int device){
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, device));
    return deviceProp.sharedMemPerBlock;
}


void writeAllInfoToFile(const char *filename, int device){

    const char* dirname = "par_hw_info/"; 
    
    createDirectory(dirname); 

    char path[256]; // Assuming max path length of 256 characters
    snprintf(path, sizeof(path), "%s%s", dirname, filename);
    FILE *file = fopen(path, "w");
    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }

    int dev, driverVersion = 0, runtimeVersion = 0;

    dev = device;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    fprintf(file, "Device %d: \"%s\"\n", dev, deviceProp.name);

    CHECK(cudaDriverGetVersion(&driverVersion));
    CHECK(cudaRuntimeGetVersion(&runtimeVersion));

    fprintf(file, "CUDA Driver Version / Runtime Version %d.%d / %d.%d\n",
    driverVersion/1000, (driverVersion%100)/10,
    runtimeVersion/1000, (runtimeVersion%100)/10);

    fprintf(file, "CUDA Capability Major/Minor version number: %d.%d\n",
    deviceProp.major, deviceProp.minor);

    fprintf(file, "Total amount of global memory: %.2f GBytes (%llu bytes)\n",
    (float)deviceProp.totalGlobalMem/(pow(1024.0,3)),
    (unsigned long long) deviceProp.totalGlobalMem);

    fprintf(file, "GPU Clock rate: %.0f MHz (%0.2f GHz)\n",
    deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

    fprintf(file, "Memory Clock rate: %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);

    fprintf(file, "Memory Bus Width: %d-bit\n", deviceProp.memoryBusWidth);

    if (deviceProp.l2CacheSize) {
        fprintf(file, "L2 Cache Size: %d bytes\n", deviceProp.l2CacheSize);
    }

    fprintf(file, "Max Texture Dimension Size (x,y,z) "
    " 1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n",
    deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
    deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0], 
    deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);

    fprintf(file, "Max Layered Texture Size (dim) x layers 1D=(%d) x %d, 2D=(%d,%d) x %d\n",
    deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1],
    deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
    deviceProp.maxTexture2DLayered[2]);

    fprintf(file, "Total amount of constant memory: %lu bytes\n", deviceProp.totalConstMem);
    
    fprintf(file, "Total amount of shared memory per block: %lu bytes\n", deviceProp.sharedMemPerBlock);
    
    fprintf(file, "Total number of registers available per block: %d\n",  deviceProp.regsPerBlock);
    
    fprintf(file, "Warp size: %d\n", deviceProp.warpSize);
    
    fprintf(file, "Maximum number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
    
    fprintf(file, "Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    
    fprintf(file, "Maximum sizes of each dimension of a block: %d x %d x %d\n",
    deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    
    fprintf(file, "Maximum sizes of each dimension of a grid: %d x %d x %d\n",
    deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    
    fprintf(file, "Maximum memory pitch: %lu bytes\n", deviceProp.memPitch);

    fprintf(file, "\n\n");

    fprintf(file, "Compiler information:\n");
    char* compilerInfo = getCompilerInfo();
    if (compilerInfo != NULL) {
        fprintf(file, "%s\n", compilerInfo);
        free(compilerInfo); // Free the memory allocated for the string
    }

    fprintf(file, "nvcc information:\n");
    char* nvccInfo = getNVCCInfo();
    if (nvccInfo != NULL) {
        fprintf(file, "%s\n", nvccInfo);
        free(nvccInfo); // Free the memory allocated for the string
    }

    fprintf(file, "Operating System information:\n");
    char* osInfo = getOSInfo();
    if (osInfo != NULL) {
        fprintf(file, "%s\n", osInfo);
        free(osInfo); // Free the memory allocated for the string
    }

    fprintf(file, "\n\n");


    // Get system information
    struct sysinfo sys_info;
    if (sysinfo(&sys_info) != 0) {
        printf("Error getting system information.\n");
        fclose(file);
        return;
    }

    // Write hardware specification to file
    fprintf(file, "System Information:\n");
    fprintf(file, "--------------------\n");
    fprintf(file, "Total RAM: %lu MB\n", sys_info.totalram / (1024 * 1024));
    fprintf(file, "Free RAM: %lu MB\n", sys_info.freeram / (1024 * 1024));
    fprintf(file, "Total Swap: %lu MB\n", sys_info.totalswap / (1024 * 1024));
    fprintf(file, "Free Swap: %lu MB\n", sys_info.freeswap / (1024 * 1024));
    fprintf(file, "Number of procs: %d\n", sys_info.procs);


    fprintf(file, "\nCPU Information:\n");
    fprintf(file, "----------------\n");

    FILE *cpuinfo = fopen("/proc/cpuinfo", "r");
    if (cpuinfo == NULL) {
        printf("Error opening CPU info file.\n");
        fclose(file);
        return;
    }

    char line[256];
    while (fgets(line, sizeof(line), cpuinfo)) {
        fputs(line, file);
    }

    fclose(cpuinfo);
    fclose(file);

    printf("\nHardware specification has been written to %s\n\n", path);
}


int nearestPowerOfTwo(int n) {
    if(n == 0){         // condition to handle 0 as input (useful for workers default calculation with too big alpha factor and small dataset)  
        return 1;
    }
    int power = 1;
    while (power <= n) {
        power *= 2;
    }
    return power / 2;
}


#endif