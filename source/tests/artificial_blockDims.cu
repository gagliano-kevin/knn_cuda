#include "../../include/cuda_functions.h"
#include <math.h>

int main(int argc, char** argv) {

    // Selection of best device
    int device = setBestDevice();
    if (device == -1){
        printf("Kernel launch abort\n");
        return -1;
    }

    printf("Executing file: %s\n\n", __FILE__);

    int k = 10; 
    int metric = 1;                                                                             // Euclidean distance
    int exp = 4;                                                                                // Power for Minkowski distance (not used in this case)
    int trainSize = 1000;                                                                       // Size of the training set
    int testSize = 100;                                                                         // Size of the test set
    int mean = 10;                                                                              // Mean value for data generation
    int num_features = 10;                                                                      // Number of features (and classes)
    int num_classes = num_features;                                                             // Number of classes
    int alpha = 2;                                                                              // Default alpha value (regulates the number of threads in a block for the knn kernel)
    int dimx = 2;                                                                               // Initial block dimension
    double exeTimes[5];                                                                         // Execution times for each block dimension

    // Loop over different block dimensions (2, 4, 8, 16, 32 if no argv[1] and argv[2] are provided)
    for(dimx = 2; dimx <= 32; dimx *= 2){

        int dimy = dimx;                                                                                        // Square blocks                      

        // Pointer to memory for data and labels
        double *trainData;
        int *trainLabels;
        double *testData;
        int *testLabels;

        generateData(trainSize, num_features, &trainData, &trainLabels, mean);                                  // Generate training set
        generateData(testSize, num_features, &testData, &testLabels, mean);                                     // Generate test set

        // Host memory allocation
        double *distances = (double *)malloc(trainSize * testSize * sizeof(double));
        int *trainIndexes = (int *)malloc(trainSize * testSize * sizeof(int));
        int *predictions = (int *)malloc(testSize * sizeof(int));

        createTrainIndexes(trainIndexes, testSize, trainSize);                                                  // Create training set indexes for each test set element

        // Pointers to device memory for data, distances, indexes, predictions and labels
        double *d_trainData, *d_testData, *d_distances;
        int *d_trainIndexes, *d_predictions, *d_trainLabels;

        // Device memory allocation for the distances kernel (knnDistances) 
        cudaMalloc(&d_trainData, trainSize * num_features * sizeof(double));
        cudaMalloc(&d_testData, testSize * num_features * sizeof(double));
        cudaMalloc(&d_distances, trainSize * testSize * sizeof(double));

        // Copy data from host to device
        cudaMemcpy(d_trainData, trainData, trainSize * num_features * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_testData, testData, testSize * num_features * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemset(d_distances, 0, trainSize * testSize * sizeof(double));                                      // Initialize distances matrix with 0

        // User defined block dimensions (if provided must be within the maximum number of threads per block)
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
        double avgKnnDistElaps = 0.0;                                                                           // Average elapsed time for distances computation
        for(int i = 1; i <= 5; i++){                                                                            // 5 iterations for average time
            double knnDistStart = cpuSecond();
            knnDistances<<< grid, block >>>(d_trainData, d_testData, d_distances, trainSize, testSize, metric, exp, num_features);
            cudaDeviceSynchronize();                                                                            //forcing synchronous behavior
            double knnDistElaps = cpuSecond() - knnDistStart;
            avgKnnDistElaps += knnDistElaps;
        }
        avgKnnDistElaps /= 5;

        // Free device memory (to keep available global memory as high as possible for the subsequent memory allocation)
        cudaFree(d_trainData);
        cudaFree(d_testData);

        // Device memory allocation for the knn kernel
        cudaMalloc(&d_trainIndexes, trainSize * testSize * sizeof(int));
        cudaMalloc(&d_predictions, testSize * sizeof(int));
        cudaMalloc(&d_trainLabels, trainSize * sizeof(int));

        // Copy data from host to device
        cudaMemcpy(d_trainIndexes, trainIndexes, trainSize * testSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_trainLabels, trainLabels, trainSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_predictions, 0, testSize * sizeof(int));                                                   // Initialize predictions array with 0

        // User defined alpha value (if provided must be within the range [2, 32])
        if(argc > 3){
            if (atoi(argv[3]) >= alpha && atoi(argv[3]) <= 32){         
                alpha = atoi(argv[3]);
            } else {
                printf("Invalid alpha value, alpha must be in range [2, 32]. Using default value: %d\n\n", alpha);
            }
        }

        int beta = 4;                                                                                           // Default beta value
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

        // User defined workers value (if provided must be within the range [1, workers])
        if(argc > 5){
                if(atoi(argv[5]) >= 1 && atoi(argv[5]) <= workers){
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
        double avgKnnElaps = 0.0;                                                                               // Average elapsed time for knn computation
        for(int i = 1; i <= 5; i++){                                                                            // 5 iterations for average time
            double knnStart = cpuSecond();
            knn<<< gridDim, blockDim, sharedMemorySize>>>(d_distances, trainSize, d_trainIndexes, k, d_predictions, d_trainLabels, index, alpha, beta, num_classes);
            cudaDeviceSynchronize();                                                                            // Forcing synchronous behavior
            double knnElaps = cpuSecond() - knnStart;
            avgKnnElaps += knnElaps;
        }
        avgKnnElaps /= 5;
 
        cudaMemcpy(predictions, d_predictions, testSize * sizeof(int), cudaMemcpyDeviceToHost);                 // Copy predictions from device to host
        int errorCount = checkResult(testLabels, predictions, testSize);                                        // Check the number of errors in the predictions

        // kernels dimensions (needed for the results file)
        unsigned int distDim[4] = {grid.x, grid.y, block.x, block.y};
        unsigned int predDim[4] = {gridDim.x, gridDim.y, blockDim.x, blockDim.y};

        // Print results to file
        appendResultsToFile(errorCount, testSize, "artificial_blockDims_cu.txt", "artificial_blockDims/", trainSize, num_features, k, metric, exp, distDim, predDim, workers, alpha, beta, avgKnnDistElaps, avgKnnElaps, sharedMemorySize, maxSharedMemory, sharedWorkers);

        exeTimes[(int)(log2((double)dimx) - 1)] = avgKnnElaps + avgKnnDistElaps;                                // Store the execution time for the current alpha value

        // Free device memory
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

        cudaDeviceReset();                                                                                      // Reset the device for the next iteration
    }

    exeTimeToFile("artificial_blockDims_csv.txt", "artificial_blockDims/", exeTimes, 5);

    return 0;
}