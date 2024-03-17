#include "../../include/cuda_functions.h"



int main(int argc, char** argv) {

    // Selection of best device
    int device = setBestDevice();
    if (device == -1){
        printf("Kernel launch abort\n");
        return -1;
    }

    printf("Executing file: %s\n\n", __FILE__);
    
    int num_features = 10; // Number of features (and classes)

    for(num_features = 10; num_features <= 100; num_features += 10){
    
        int k = 10; 
        int metric = 1; // Euclidean distance
        int exp = 4; // Power for Minkowski distance (not used in this case)


        int trainSize = 10000; // Size of the dataset
        int testSize = 1000; // Size of the dataset
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
        
    
        // Set squared maximum dimensions as default
        int dimx = (int)sqrt(getMaxThreadsPerBlock(device)); 
        int dimy = dimx;

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

        cudaMemset(d_distances, 0, trainSize * testSize * sizeof(double)); // initialize distances matrix with 0
        
        // Set cache configuration for the kernel -> prefer 48KB L1 cache and 16KB shared memory
        cudaFuncSetCacheConfig(knnDistances, cudaFuncCachePreferL1);


        double knnDistStart = cpuSecond();
        knnDistances<<< grid, block >>>(d_trainData, d_testData, d_distances, trainSize, testSize, metric, exp, num_features);
        cudaDeviceSynchronize();        //forcing synchronous behavior
        double knnDistElaps = cpuSecond() - knnDistStart;
        

        int alpha = 2;  // default
        if(argc > 3){
            if (atoi(argv[3]) >= alpha && atoi(argv[3]) <= 32){         // alpha limited up to 32
                alpha = atoi(argv[3]);
            } else {
                printf("Invalid alpha value, alpha must be in range [2, 32]. Using default value: %d\n\n", alpha);
            }
        }

        int beta = 4;   // default
        if(argc > 4){
            if (atoi(argv[4]) >= beta && atoi(argv[4]) <= 32){         // beta limited up to 32
                beta = atoi(argv[4]);
            } else {
                printf("Invalid beta value, beta must be in range [4, 32]. Using default value: %d\n\n", beta);
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
            if (atoi(argv[5]) < workers && atoi(argv[5]) >= 1){
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
        knn<<< gridDim, blockDim, sharedMemorySize>>>(d_distances, trainSize, d_trainIndexes, k, d_predictions, d_trainLabels, index, alpha, beta, num_classes);
        cudaDeviceSynchronize();        //forcing synchronous behavior
        double knnElaps = cpuSecond() - knnStart;

        cudaMemcpy(distances, d_distances, trainSize * testSize * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(trainIndexes, d_trainIndexes, trainSize * testSize * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(predictions, d_predictions, testSize * sizeof(int), cudaMemcpyDeviceToHost);

        //check device results
        int errorCount = checkResult(testLabels, predictions, testSize);

        // kernels dimensions
        unsigned int distDim[4] = {grid.x, grid.y, block.x, block.y};
        unsigned int predDim[4] = {gridDim.x, gridDim.y, blockDim.x, blockDim.y};

        // Print results to file
        appendResultsToFile(errorCount, testSize, "artificial_features.txt", "artificial_features/", trainSize, num_features, k, metric, exp, distDim, predDim, workers, alpha, beta, knnDistElaps, knnElaps, sharedMemorySize, maxSharedMemory, sharedWorkers);


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
    }


    return 0;
}