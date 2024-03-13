#include "../include/cuda_functions.h"
#include "../include/iris_functions.h"

int main(int argc, char** argv) {

    // Selection of best device
    int device = setBestDevice();
    if (device == -1){
        printf("Kernel launch abort\n");
        return -1;
    }

    int k = 5; // k = 5
    int metric = 3; // Metric distance
    int exp = 4; // Power for Minkowski distance

    IrisData *iris_data;
    int trainSize;

    // Read the Iris dataset
    if (readIrisDataset("../datasets/Iris.csv", &iris_data, &trainSize) != 0) {
        fprintf(stderr, "Error reading Iris dataset\n");
        return 1;
    }

    int testSize = trainSize/3;

    double *trainData = (double *)malloc(trainSize * FEATURES * sizeof(double));
    int *trainLabels = (int *)malloc(trainSize * sizeof(int));

    createTrainingSet(iris_data, trainData, trainLabels, trainSize);

    // Test set (1/3 of training set, balanced over classes -> 17,17,16)
    size_t testDataSize = (trainSize / 3) * FEATURES * sizeof(double);
    size_t testLabelsSize = (trainSize / 3) * sizeof(int);
    
    double *testData = (double *)malloc(testDataSize);
    int *testLabels = (int *)malloc(testLabelsSize);
    
    createTestSet(trainData, testData, trainLabels, testLabels, trainSize);

    double *distances = (double *)malloc(trainSize * testSize * sizeof(double));

    int *trainIndexes = (int *)malloc(trainSize * testSize * sizeof(int));
    int *predictions = (int *)malloc(testSize * sizeof(int));


    createTrainIndexes(trainIndexes, testSize, trainSize);

    // device
    double *d_trainData, *d_testData, *d_distances;
    int *d_trainIndexes, *d_predictions, *d_trainLabels;

    cudaMalloc(&d_trainData, trainSize * FEATURES * sizeof(double));
    cudaMalloc(&d_testData, testSize * FEATURES * sizeof(double));
    cudaMalloc(&d_distances, trainSize * testSize * sizeof(double));
    cudaMalloc(&d_trainIndexes, trainSize * testSize * sizeof(int));
    cudaMalloc(&d_predictions, testSize * sizeof(int));
    cudaMalloc(&d_trainLabels, trainSize * sizeof(int));

    cudaMemcpy(d_trainData, trainData, trainSize * FEATURES * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_testData, testData, testSize * FEATURES * sizeof(double), cudaMemcpyHostToDevice);
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

    cudaMemset(d_distances, 0, trainSize * testSize * sizeof(double)); // initialize distances matrix with 0
    
    // Set cache configuration for the kernel -> prefer 48KB L1 cache and 16KB shared memory
    cudaFuncSetCacheConfig(knnDistances, cudaFuncCachePreferL1);

    double knnDistStart = cpuSecond();
    knnDistances<<< grid, block >>>(d_trainData, d_testData, d_distances, trainSize, testSize, metric, exp, FEATURES);
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

    int beta = 4;
    if(argc > 5){
        beta = atoi(argv[5]);
    }
    int sharedWorkers = (int)(blockDim.x / alpha);
    int additionalMemory = k * sharedWorkers * (sizeof(double) + sizeof(int));  // blockDim.x/alpha is the number of workers in 2^ iteration (first in shared memory)

    int sharedMemorySize = (k * blockDim.x) * (sizeof(double) + sizeof(int)) + additionalMemory; 

    int index = k * (blockDim.x + sharedWorkers); // starting index for trainIndexes in shared memory 

    double knnSortStart = cpuSecond();
    knnSortPredict<<< gridDim, blockDim, sharedMemorySize>>>(d_distances, trainSize, d_trainIndexes, k, d_predictions, d_trainLabels, index, alpha, beta, CLASSES);
    cudaDeviceSynchronize();        //forcing synchronous behavior
    double knnSortElaps = cpuSecond() - knnSortStart;

    cudaMemcpy(distances, d_distances, trainSize * testSize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(trainIndexes, d_trainIndexes, trainSize * testSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(predictions, d_predictions, testSize * sizeof(int), cudaMemcpyDeviceToHost);

    //printDataSet(trainData, trainLabels, trainSize);

    //printDistances(distances, testSize, trainSize);

    //printTrainIndexes(trainIndexes, testSize, trainSize);


    //check device results
    int errorCount = checkResult(testLabels, predictions, testSize);

    // kernels dimensions
    unsigned int distDim[4] = {grid.x, grid.y, block.x, block.y};
    unsigned int predDim[4] = {gridDim.x, gridDim.y, blockDim.x, blockDim.y};

    // Write results and device info to file
    writeResultsToFile(testLabels, predictions, errorCount, testSize, "par_results.txt", trainSize, FEATURES, k, metric, exp, distDim, predDim, workers, alpha, beta, knnDistElaps, knnSortElaps); 
    //writeDeviceInfo("device_info.txt", device);
    writeAllInfoToFile("all_HW_info.txt", device);

    // Free device memory
    cudaFree(d_trainData);
    cudaFree(d_testData);
    cudaFree(d_distances);
    cudaFree(d_trainIndexes);
    cudaFree(d_predictions);
    cudaFree(d_trainLabels);


    // Free host memory
    free(iris_data);
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
