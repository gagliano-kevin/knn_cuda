// ******************* Instance of knn_parallel.cu on artificial Data (KNN v.3 -> BEST VERSION) ***********************


//nvcc knn_artificialData.cu -o artificial
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <time.h>


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


double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
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
extern "C" __global__ void knnSortPredict(double *distances, int trainSize, int *indexes, int k, int *predictions, int *trainLabels, int sharedMemoryIdx, int alpha, int beta, int classes) {
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
        //int classCounts[classes] = {0};
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



void writeResultsToFile(int * trainLabels, int *results, int errorCount, int testSize, const char *filename, int trainSize, int features, int k, int metric, int exp, unsigned int *distDim, unsigned int *predDim, int workers, int alpha, int beta, double kernelTime1, double kernelTime2) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }

    fprintf(file, "Kernel launch information:\n");
    fprintf(file, "Grid dimension in knnDistances kernel: %u , %u\n", distDim[0], distDim[1]);
    fprintf(file, "Block dimension in knnDistances kernel: %u , %u\n", distDim[2], distDim[3]);
    fprintf(file, "Grid dimension in knnSortPredict kernel: %u , %u\n",predDim[0], predDim[1]);
    fprintf(file, "Block dimension in knnSortPredict kernel: %u , %u\n", predDim[2], predDim[3]);
    fprintf(file, "Number of workers: %d\n", workers);
    fprintf(file, "Factor alpha: %d\n", alpha);
    fprintf(file, "Factor beta: %d\n", beta);
    fprintf(file, "knnDistances execution time %f sec\n", kernelTime1);
    fprintf(file, "knnSortPredict execution time %f sec\n", kernelTime2);

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
        printf("Setting Device %d : \"%s\"\n", maxDevice, best_prop.name);
        return maxDevice;
    } else {
        printf("Detected only one CUDA Device ...\n");
        int dev = 0;
        CHECK(cudaSetDevice(dev));
        cudaDeviceProp deviceProp;
        CHECK(cudaGetDeviceProperties(&deviceProp, dev));
        printf("Setting Device %d: \"%s\"\n", dev, deviceProp.name);
        return dev;
    }
}


int checkResult(int *labels, int *predictions, const int N){
    int errorCount = 0;
    for (int i=0; i<N; i++){
        if(labels[i] != predictions[i]){
            errorCount++;
        }
    }
    printf("Number of prediction errors: %d\n\n", errorCount);
    return errorCount;
}


void printDataSet(double *trainData, int *trainLabels, int trainSize, int num_features){
    for(int i = 0; i < trainSize; i++){
        printf("Data[%d]", i);
        for(int j = 0; j < num_features; j++){
            int idx = i * num_features + j;
            printf("%9.3f", trainData[idx]);
        }
        printf(" -> label: %d\n\n", trainLabels[i]);
    }
}



void createTrainIndexes(int *trainIndexes, int testSize, int trainSize){
    for(int i = 0; i < testSize; i++){
        for(int j = 0; j < trainSize; j++){
            int index = i * trainSize + j; 
            trainIndexes[index] = j;
        }
    }
}


void printTrainIndexes(int *trainIndexes, int testSize, int trainSize){
    printf("\nTrain Indexes :\n");
    for(int i = 0; i < testSize; i++){
        for(int j = 0; j < trainSize; j++){
            int idx = i * trainSize + j; 
            printf("%d \t", trainIndexes[idx]);
        }
        printf("\n\n\n");
    }
}


void printDistances(double *distances, int testSize, int trainSize){
    printf("\nDistances :\n");
    for(int i = 0; i < testSize; i++){
        for(int j = 0; j < trainSize; j++){
            int idx = trainSize * i + j;
            printf("%3.3f\t", distances[idx]);
        }
        printf("\n\n\n");
    }
}


// Function to generate a Dataset
void generateData(int size, int num_features, double **data, int **labels, double mean) {
    // Allocate memory for data and labels
    *data = (double *)malloc(size * num_features * sizeof(double));
    *labels = (int *)malloc(size * sizeof(int));
    
    // Generate training data
    double noise = 0.1; // Adjust this value to control noise level
    
    srand(time(NULL)); // Seed for random number generation
    
    for (int i = 0; i < size; i++) {
        int class_index = i % num_features;
        
        // Fill data vector with noise
        for (int j = 0; j < num_features; j++) {
            if (j == class_index) {
                // Generate value for class component as sum of mean value and noise
                (*data)[i * num_features + j] = mean + ((double)rand() / RAND_MAX) * noise;
            } else {
                // Other components are noise
                (*data)[i * num_features + j] = ((double)rand() / RAND_MAX) * noise;
            }
        }
        
        // Assign label
        (*labels)[i] = class_index;
    }
}



int main(int argc, char** argv) {

    // Selection of best device
    int device = setBestDevice();
    if (device == -1){
        printf("Kernel launch abort\n");
        return -1;
    }

    int k = 10; 
    int metric = 3; // Metric distance
    int exp = 4; // Power for Minkowski distance


    int trainSize = 1000; // Size of the dataset
    int testSize = 100; // Size of the dataset
    int num_features = 10; // Number of features (and classes)
    int num_classes = num_features; // Number of classes
    int mean = 10; // Mean value for class component
    
    // pointer to memory for data and labels
    double *trainData;
    int *trainLabels;
    double *testData;
    int *testLabels;
    
    // Generate training set
    generateData(trainSize, num_features, &trainData, &trainLabels, mean);
    // Generate test set
    generateData(testSize, num_features, &testData, &testLabels, mean);

    // Host memory allocation
    double *distances = (double *)malloc(trainSize * testSize * sizeof(double));
    int *trainIndexes = (int *)malloc(trainSize * testSize * sizeof(int));
    int *predictions = (int *)malloc(testSize * sizeof(int));


    createTrainIndexes(trainIndexes, testSize, trainSize);

    // device memory allocation
    double *d_trainData, *d_testData, *d_distances;
    int *d_trainIndexes, *d_predictions, *d_trainLabels;

    cudaMalloc(&d_trainData, trainSize * num_features * sizeof(double));
    cudaMalloc(&d_testData, testSize * num_features * sizeof(double));
    cudaMalloc(&d_distances, trainSize * testSize * sizeof(double));
    cudaMalloc(&d_trainIndexes, trainSize * testSize * sizeof(int));
    cudaMalloc(&d_predictions, testSize * sizeof(int));
    cudaMalloc(&d_trainLabels, trainSize * sizeof(int));

    cudaMemcpy(d_trainData, trainData, trainSize * num_features * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_testData, testData, testSize * num_features * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_trainIndexes, trainIndexes, trainSize * testSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_trainLabels, trainLabels, trainSize * sizeof(int), cudaMemcpyHostToDevice);
    

    int dimx = 32;      //default
    int dimy = 32;      //default
    if(argc > 2){
        dimx = atoi(argv[1]);
        dimy = atoi(argv[2]);
    }
    dim3 block(dimx, dimy);
    dim3 grid((trainSize + block.x-1)/block.x, (testSize + block.y-1)/block.y);

    double knnDistStart = cpuSecond();
    knnDistances<<< grid, block >>>(d_trainData, d_testData, d_distances, trainSize, testSize, metric, exp, num_features);
    cudaDeviceSynchronize();        //forcing synchronous behavior
    double knnDistElaps = cpuSecond() - knnDistStart;
    
    int workers = 10;       // default
    if(argc > 3){
        workers = atoi(argv[3]);
    }

    dim3 gridDim(testSize, 1, 1);   // each thread block is responsible for a row of the distances matrix
    dim3 blockDim(workers, 1, 1);
    int alpha = 2;  // default
    if(argc > 4){
        alpha = atoi(argv[4]);
    }

    int beta = 4;   // default
    if(argc > 5){
        beta = atoi(argv[5]);
    }
    int sharedWorkers = (int)(blockDim.x / alpha);
    int additionalMemory = k * sharedWorkers * (sizeof(double) + sizeof(int));  // blockDim.x/alpha is the number of workers in 2^ iteration (first in shared memory)

    int sharedMemorySize = (k * blockDim.x) * (sizeof(double) + sizeof(int)) + additionalMemory; 

    int index = k * (blockDim.x + sharedWorkers); // starting index for trainIndexes in shared memory 

    double knnSortStart = cpuSecond();
    knnSortPredict<<< gridDim, blockDim, sharedMemorySize>>>(d_distances, trainSize, d_trainIndexes, k, d_predictions, d_trainLabels, index, alpha, beta, num_classes);
    cudaDeviceSynchronize();        //forcing synchronous behavior
    double knnSortElaps = cpuSecond() - knnSortStart;

    cudaMemcpy(distances, d_distances, trainSize * testSize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(trainIndexes, d_trainIndexes, trainSize * testSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(predictions, d_predictions, testSize * sizeof(int), cudaMemcpyDeviceToHost);

    //printDataSet(trainData, trainLabels, trainSize, num_features);

    //printDistances(distances, testSize, trainSize);

    //printTrainIndexes(trainIndexes, testSize, trainSize);


    //check device results
    int errorCount = checkResult(testLabels, predictions, testSize);

    // kernels dimensions
    unsigned int distDim[4] = {grid.x, grid.y, block.x, block.y};
    unsigned int predDim[4] = {gridDim.x, gridDim.y, blockDim.x, blockDim.y};

    // Write results and device info to file
    writeResultsToFile(testLabels, predictions, errorCount, testSize, "par_results_artificial.txt", trainSize, num_features, k, metric, exp, distDim, predDim, workers, alpha, beta, knnDistElaps, knnSortElaps); 
    writeDeviceInfo("device_info.txt", device);

    // Free device memory
    cudaFree(d_trainData);
    cudaFree(d_testData);
    cudaFree(d_distances);
    cudaFree(d_trainIndexes);
    cudaFree(d_predictions);
    cudaFree(d_trainLabels);


    // Free host memory
    free(trainData);
    free(trainLabels);
    free(testData);
    free(testLabels);
    free(distances);
    free(trainIndexes);
    free(predictions);


    //reset device
    cudaDeviceReset();


    return 0;
}
