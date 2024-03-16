#include "../include/cuda_functions.h"
#include "../include/diabetes_functions.h"


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

    Row *dataset;
    int trainSize;
    int testSize;

    // TRAINING DATA
    if (readCSV("../datasets/diabetes_training.csv", &dataset, &trainSize) != 1) {
        printf("Error reading CSV file.\n");
        return 1;
    }

    // Allocate memory for trainData
    double *trainData = (double *)malloc(trainSize * FEATURES * sizeof(double));
    if (trainData == NULL) {
        printf("Error allocating memory.\n");
        free(dataset);
        return 1;
    }

    // Allocate memory for train labels
    int *trainLabels = (int *)malloc(trainSize * sizeof(int));
    if (trainLabels == NULL) {
        printf("Error allocating memory.\n");
        free(dataset);
        free(trainData);
        return 1;
    }

    // Training data extraction
    extractData(dataset, trainData, trainLabels, trainSize);
    //printDataSet(trainData, trainLabels, numRows);

    
    // TEST DATA
    if (readCSV("../datasets/diabetes_testing.csv", &dataset, &testSize) != 1) {
        printf("Error reading CSV file.\n");
        return 1;
    }

    // Allocate memory for testData
    double *testData = (double *)malloc(testSize * FEATURES * sizeof(double));
    if (testData == NULL) {
        printf("Error allocating memory.\n");
        free(dataset);
        return 1;
    }

    // Allocate memory for test labels
    int *testLabels = (int*)malloc(testSize * sizeof(int));
    if (testLabels == NULL) {
        printf("Error allocating memory.\n");
        free(dataset);
        free(testData);
        return 1;
    }

    // Test data extraction
    extractData(dataset, testData, testLabels, testSize);
    //printDataSet(testData, testLabels, testSize);


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
    

    // Set squared maximum dimensions as default
    int dimx = (int)sqrt(getMaxThreadsPerBlock(device)); 
    int dimy = dimx;

    if(argc > 2){
        if (atoi(argv[1]) * atoi(argv[2]) <= getMaxThreadsPerBlock(device)){
            dimx = atoi(argv[1]);
            dimy = atoi(argv[2]);
        } else {
            printf("Invalid dimensions. Maximum number of threads per block is %d\n", getMaxThreadsPerBlock(device));
            printf("Using default dimensions: %d x %d\n\n", dimx, dimy);
        }
    }
    dim3 block(dimx, dimy);
    dim3 grid((trainSize + block.x-1)/block.x, (testSize + block.y-1)/block.y);

    cudaMemset(d_distances, 0, trainSize * testSize * sizeof(double)); // initialize distances matrix with 0
    
    // Set cache configuration for the kernel -> prefer 48KB L1 cache and 16KB shared memory
    cudaFuncSetCacheConfig(knnDistances, cudaFuncCachePreferL1);
    
    printf("Executing file: %s\n\n", __FILE__);

    double knnDistStart = cpuSecond();
    knnDistances<<< grid, block >>>(d_trainData, d_testData, d_distances, trainSize, testSize, metric, exp, FEATURES);
    cudaDeviceSynchronize();        //forcing synchronous behavior
    double knnDistElaps = cpuSecond() - knnDistStart;
    
    int alpha = 2;  // default
    if(argc > 3){
        if (atoi(argv[3]) >= alpha){
            alpha = atoi(argv[3]);
        } else {
            printf("Invalid alpha value. Using default value: %d\n\n", alpha);
        }
    }

    int beta = 4;   // default
    if(argc > 4){
        if (atoi(argv[4]) >= beta){
            beta = atoi(argv[4]);
        } else {
            printf("Invalid beta value. Using default value: %d\n\n", beta);
        }
    }

    int maxSharedMemory = getSharedMemoryPerBlock(device);
    int itemSize = sizeof(double) + sizeof(int);
    int workers = maxSharedMemory/(k * itemSize * (1.5 + 1/alpha));       // default (maxization of shared memory usage)
    workers = nearestPowerOfTwo(workers);
    if (workers > (int)trainSize/(alpha*k)){
        workers = nearestPowerOfTwo((int)trainSize/(alpha*k));           // new default in case of too many workers (small dataset)
    }

    if(argc > 5){
        if (atoi(argv[5]) < workers){
            workers = atoi(argv[5]);
        } else {
            printf("Invalid workers value. Using default value: %d\n\n", workers);
        }
    }

    dim3 gridDim(testSize, 1, 1);   // each thread block is responsible for a row of the distances matrix
    dim3 blockDim(workers, 1, 1);

    int sharedWorkers = (int)(blockDim.x / alpha);
    int additionalMemory = k * sharedWorkers * (sizeof(double) + sizeof(int));  // blockDim.x/alpha is the number of workers in 2^ iteration (first in shared memory)

    int sharedMemorySize = (k * blockDim.x) * (sizeof(double) + sizeof(int)) + additionalMemory; 

    int index = k * (blockDim.x + sharedWorkers); // starting index for trainIndexes in shared memory 

    double knnStart = cpuSecond();
    knn<<< gridDim, blockDim, sharedMemorySize>>>(d_distances, trainSize, d_trainIndexes, k, d_predictions, d_trainLabels, index, alpha, beta, CLASSES);
    cudaDeviceSynchronize();        //forcing synchronous behavior
    double knnElaps = cpuSecond() - knnStart;

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
    writeResultsToFile(testLabels, predictions, errorCount, testSize, "par_results_diabetes.txt", "par_results_diabetes/", trainSize, FEATURES, k, metric, exp, distDim, predDim, workers, alpha, beta, knnDistElaps, knnElaps, sharedMemorySize, maxSharedMemory); 
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
    free(dataset);
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
