#include "../include/cuda_functions.h"
#include "../include/iris_functions.h"

int main(int argc, char** argv) {

    // Selection of best device
    int device = setBestDevice();
    if (device == -1){
        printf("Kernel launch abort\n");
        return -1;
    }

    printf("Executing file: %s\n\n", __FILE__);

    int k = 10; 
    int metric = 1;                                                                                             // Euclidean distance
    int exp = 4;                                                                                                // Power for Minkowski distance (not used in this case)
    double exeTimes[5] = {0.0, 0.0, 0.0, 0.0, 0.0};

    IrisData *iris_data;
    int trainSize;

    // Read the Iris dataset
    if (readIrisDataset("../datasets/Iris.csv", &iris_data, &trainSize) != 0) {
        fprintf(stderr, "Error reading Iris dataset\n");
        return 1;
    }

    int testSize = trainSize/3;                                                                                 // Test set size (1/3 of training set)

    // Training set
    double *trainData = (double *)malloc(trainSize * FEATURES * sizeof(double));
    int *trainLabels = (int *)malloc(trainSize * sizeof(int));
    createTrainingSet(iris_data, trainData, trainLabels, trainSize);

    // Test set (1/3 of training set, balanced over classes -> 17,17,16)
    int testDataSize = (trainSize / 3) * FEATURES * sizeof(double);
    int testLabelsSize = (trainSize / 3) * sizeof(int);
    double *testData = (double *)malloc(testDataSize);
    int *testLabels = (int *)malloc(testLabelsSize);
    createTestSet(trainData, testData, trainLabels, testLabels, trainSize);

    // Host memory allocation
    double *distances = (double *)malloc(trainSize * testSize * sizeof(double));
    int *trainIndexes = (int *)malloc(trainSize * testSize * sizeof(int));
    int *predictions = (int *)malloc(testSize * sizeof(int));

    createTrainIndexes(trainIndexes, testSize, trainSize);                                                      // Create training set indexes for each test set element

    // Pointers to device memory for data, distances, indexes, predictions and labels
    double *d_trainData, *d_testData, *d_distances;
    int *d_trainIndexes, *d_predictions, *d_trainLabels;

    // Device memory allocation 
    cudaMalloc(&d_trainData, trainSize * FEATURES * sizeof(double));
    cudaMalloc(&d_testData, testSize * FEATURES * sizeof(double));
    cudaMalloc(&d_distances, trainSize * testSize * sizeof(double));
    cudaMalloc(&d_trainIndexes, trainSize * testSize * sizeof(int));
    cudaMalloc(&d_predictions, testSize * sizeof(int));
    cudaMalloc(&d_trainLabels, trainSize * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_trainData, trainData, trainSize * FEATURES * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_testData, testData, testSize * FEATURES * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_trainIndexes, trainIndexes, trainSize * testSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_trainLabels, trainLabels, trainSize * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMemset(d_distances, 0, trainSize * testSize * sizeof(double));                                              // Initialize distances matrix to 0    
    cudaMemset(d_predictions, 0, testSize * sizeof(int));                                                           // Initialize predictions array to 0

    // Set squared maximum dimensions as default for the blocks
    int dimx = (int)sqrt(getMaxThreadsPerBlock(device)); 
    int dimy = dimx;

    // User defined block dimensions (if provided must be whitin the maximum number of threads per block)
    if(argc > 2){
        if (atoi(argv[1]) * atoi(argv[2]) <= getMaxThreadsPerBlock(device)){
            dimx = atoi(argv[1]);
            dimy = atoi(argv[2]);
        } else {
            printf("Invalid block dimensions for distances computation. Maximum number of threads per block is %d\n", getMaxThreadsPerBlock(device));
            printf("Using default dimensions: %d x %d\n\n", dimx, dimy);
        }
    }
    dim3 block(dimx, dimy);
    dim3 grid((trainSize + block.x-1)/block.x, (testSize + block.y-1)/block.y);
    
    // Set cache configuration for the kernel -> prefer 48KB L1 cache and 16KB shared memory
    cudaFuncSetCacheConfig(knnDistances, cudaFuncCachePreferL1);

    // Distances computation
    double avgKnnDistElaps = 0.0;                                                                               // Average elapsed time for distances computation                                          
    for(int i = 1; i <= 5; i++){                                                                                // 5 iterations for time average                              
        double knnDistStart = cpuSecond();
        knnDistances<<< grid, block >>>(d_trainData, d_testData, d_distances, trainSize, testSize, metric, exp, FEATURES);
        cudaDeviceSynchronize();                                                                                // Forcing synchronous behavior
        exeTimes[i-1] = (cpuSecond() - knnDistStart);
        avgKnnDistElaps += exeTimes[i-1];
    }
    avgKnnDistElaps /= 5;

    
    int alpha = 2;                                                                                              // Default alpha value 
    // User defined alpha value (if provided must be within the range [2, 32])
    if(argc > 3){
        if (atoi(argv[3]) >= alpha && atoi(argv[3]) <= 32){         
            alpha = atoi(argv[3]);
        } else {
            printf("Invalid alpha value, alpha must be in range [2, 32]. Using default value: %d\n\n", alpha);
        }
    }

    int beta = 4;                                                                                           // default beta value
    // User defined beta value (if provided must be within the range [4, 32])
    if(argc > 4){
        if (atoi(argv[4]) >= beta && atoi(argv[4]) <= 32){         
            beta = atoi(argv[4]);
        } else {
            printf("Invalid beta value, beta must be in range [4, 32]. Using default value: %d\n\n", beta);
        }
    }

    int maxSharedMemory = getSharedMemoryPerBlock(device);                                                  // Maximum shared memory per block
    int itemSize = sizeof(double) + sizeof(int);                                                            // Size of each item in shared memory (distance + index)
    int workers = maxSharedMemory/(k * itemSize * (1.5 + 1/alpha));                                         // Default number of threads in a block (maxization of shared memory usage)
    workers = nearestPowerOfTwo(workers);                                                                   // Round the workers number to the nearest power of two
    if (workers > (int)trainSize/(alpha*k)){                                                                // In case of too many workers (small dataset)
        workers = nearestPowerOfTwo((int)trainSize/(alpha*k));                                              // Set new default value  
    }

    // User defined workers number (if provided must be within the range [1, workers])
    if(argc > 5){
        if (atoi(argv[5]) < workers && atoi(argv[5]) >= 1){
            workers = atoi(argv[5]);
        } else {
            printf("Invalid workers value. Using default value: %d\n\n", workers);
        }
    }

    dim3 gridDim(testSize, 1, 1);                                                                           // Each block in the grid is responsible for a row of the distances matrix
    dim3 blockDim(workers, 1, 1);                                                                           // Each thread is responsible for a subset of the corresponding row       

    int sharedWorkers = (int)(blockDim.x / alpha);                                                          // Number of workers in shared memory
    int additionalMemory = k * sharedWorkers * (sizeof(double) + sizeof(int));                              // Additional memory needed from shared workers
    int sharedMemorySize = (k * blockDim.x) * (sizeof(double) + sizeof(int)) + additionalMemory;            // Shared memory size
    int index = k * (blockDim.x + sharedWorkers);                                                           // Starting index for trainIndexes in shared memory 

    // KNN computation
    double avgKnnElaps = 0.0;                                                                               // Average elapsed time for KNN computation        
    for(int i = 1; i <= 5; i++){                                                                            // 5 iterations for time average
        double knnStart = cpuSecond();
        knn<<< gridDim, blockDim, sharedMemorySize>>>(d_distances, trainSize, d_trainIndexes, k, d_predictions, d_trainLabels, index, alpha, beta, CLASSES);
        cudaDeviceSynchronize();                                                                            // Forcing synchronous behavior
        double knnElaps = (cpuSecond() - knnStart);
        exeTimes[i-1] += knnElaps;
        avgKnnElaps += knnElaps;
    }
    avgKnnElaps /= 5;

    double exeTime = avgKnnDistElaps + avgKnnElaps;                                                         // Total execution time

    cudaMemcpy(predictions, d_predictions, testSize * sizeof(int), cudaMemcpyDeviceToHost);                 // Copy predictions from device to host
    int errorCount = checkResult(testLabels, predictions, testSize);                                        // Check the number of errors in the predictions

    // kernels dimensions 
    unsigned int distDim[4] = {grid.x, grid.y, block.x, block.y};
    unsigned int predDim[4] = {gridDim.x, gridDim.y, blockDim.x, blockDim.y};

    // Print results to file
    appendRunStatsToFile("iris_cu.txt", "iris/", exeTimes, 5);
    appendResultsToFile(errorCount, testSize, "iris_cu.txt", "iris/", trainSize, FEATURES, k, metric, exp, distDim, predDim, workers, alpha, beta, avgKnnDistElaps, avgKnnElaps, sharedMemorySize, maxSharedMemory, sharedWorkers);

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

    cudaDeviceReset();                                                                                       // Reset the device for the next iteration

    exeTimeToFile("iris_csv.txt", "iris/", &exeTime, 1);                                                     // Write execution time to csv file                                   

    return 0;
}
