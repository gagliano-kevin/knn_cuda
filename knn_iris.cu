#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <time.h>


#define MAX_LINE_SIZE 1024
#define MAX_FIELD_SIZE 128
#define NUM_FIELDS 6    // Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species

#define FILE_NAME __FILE__
#define LINE __LINE__

#define FEATURES 4
#define CLASSES 3



typedef struct {
    int id;
    float sepal_length;
    float sepal_width;
    float petal_length;
    float petal_width;
    char species[MAX_FIELD_SIZE];
} IrisData;


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
__device__ double computeDistance(double *point1, double *point2, int metric) {
    double distance = 0.0;
    if (metric == 1) { // Euclidean distance
        for (int i = 0; i < FEATURES; i++) {
            distance += (point1[i] - point2[i]) * (point1[i] - point2[i]);
        }
        distance = sqrt(distance);
    } else if (metric == 2) { // Manhattan distance
        for (int i = 0; i < FEATURES; i++) {
            distance += fabs(point1[i] - point2[i]);
        }
    } else if (metric == 3) { // Minkowski distance with p = 3
        double sum = 0.0;
        for (int i = 0; i < FEATURES; i++) {
            sum += pow(fabs(point1[i] - point2[i]), 3);
        }
        distance = pow(sum, 1.0 / 3.0);
    }
    return distance;
}


// distances matrix has size testSize x TrainSize 
__global__ void knnDistances(double *trainData, double *testData, double *distances, int trainSize, int testSize, int metric) { //testSize and trainSize could be const
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * trainSize + ix; 
    if(ix < trainSize && iy < testSize){
        distances[idx]=computeDistance(&trainData[ix * FEATURES], &testData[iy * FEATURES], metric);
    }
}


// Bubble sort on each row (trainSize) for distances and trainSet indexes + class votes
__global__ void knnSortPredict(double *distances, int trainSize, int *indexes, int k, int *predictions, int *trainLabels) {
    int row = blockIdx.x;
    int startIdx = row * trainSize;
    int endIdx = startIdx + trainSize;

    // Bubble sort 
    for (int i = startIdx; i < endIdx - 1; i++) {
        for (int j = startIdx; j < endIdx - i + startIdx - 1; j++) {
            if (distances[j] > distances[j + 1]) {
                // Swap elements (distances and indexes)
                double tempDist = distances[j];
                int tempInd = indexes[j];
                distances[j] = distances[j+1];
                indexes[j] = indexes[j+1];
                distances[j+1] = tempDist;
                indexes[j+1] = tempInd;
            }
        }
    }
    // nearest class election 
    int classCounts[CLASSES] = {0};
    for (int i = 0; i < k; i++){        //alternative (int i = 1; i <= k; i++) <- avoid to consider the first element (distance = 0.0)
        int idx = startIdx + i;
        classCounts[trainLabels[indexes[idx]]]++;
    }
    int max = 0;
    int maxClass = -1;
    for (int i = 0; i < CLASSES; i++){        
        if(classCounts[i] > max){
            maxClass = i;
        }
    }
    predictions[row] = maxClass;
}


void writeResultsToFile(int * trainLabels, int *results, int errorCount, int testSize, const char *filename, int trainSize, int features, int k, int metric, unsigned int *distDim, unsigned int *predDim, double kernelTime1, double kernelTime2) {
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
        fprintf(file, "Minkowski (p=3)\n");
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


int readIrisDataset(const char *filename, IrisData **iris_data, int *num_samples) {
    FILE *file;
    char line[MAX_LINE_SIZE];
    char *token;

    // Open the CSV file
    file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file\n");
        return 1;
    }

    // Skip the first line
    if (fgets(line, sizeof(line), file) == NULL) {
        fprintf(stderr, "Error reading file\n");
        fclose(file);
        return 1;
    }

    int count = 0;
    *num_samples = 0;
    // Count the number of lines (excluding the first)
    while (fgets(line, sizeof(line), file) != NULL) {
        (*num_samples)++;
    }

    // Allocate memory for IrisData array
    *iris_data = (IrisData *)malloc(*num_samples * sizeof(IrisData));
    if (*iris_data == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        return 1;
    }

    // Reset file pointer to the beginning
    fseek(file, 0, SEEK_SET);
    // Skip the first line
    fgets(line, sizeof(line), file);

    // Read each subsequent line of the file
    while (fgets(line, sizeof(line), file) != NULL) {
        IrisData *iris = &((*iris_data)[count]);
        int field_index = 0;

        // Tokenize each line based on the comma delimiter
        token = strtok(line, ",");
        while (token != NULL && field_index < NUM_FIELDS) {
            // Parse and store the tokenized values into appropriate data structures
            switch (field_index) {
                case 0:
                    iris->id = atoi(token);
                    break;
                case 1:
                    iris->sepal_length = atof(token);
                    break;
                case 2:
                    iris->sepal_width = atof(token);
                    break;
                case 3:
                    iris->petal_length = atof(token);
                    break;
                case 4:
                    iris->petal_width = atof(token);
                    break;
                case 5:
                token = strtok(token, "\n");
                    strcpy(iris->species, token);
                    break;
            }
            // Get the next token
            token = strtok(NULL, ",");
            field_index++;
        }
        count++;
    }

    // Close the file
    fclose(file);

    return 0;
}


// Function to map species string to class number
int mapSpeciesToClass(const char *species) {
    if (strcmp(species, "Iris-setosa") == 0) {
        return 0;
    } else if (strcmp(species, "Iris-versicolor") == 0) {
        return 1;
    } else if (strcmp(species, "Iris-virginica") == 0) {
        return 2;
    } else {
        return -1; // Unknown species
    }
}


// Training set composed of all the dataset samples
void createTrainingSet(IrisData *iris_data, double *trainData, int *trainLabels, int numSamples){
    for(int i = 0; i < numSamples; i++){
        trainData[i*4] = iris_data[i].sepal_length;
        trainData[i*4+1] = iris_data[i].sepal_width;
        trainData[i*4+2] = iris_data[i].petal_length;
        trainData[i*4+3] = iris_data[i].petal_width;
        trainLabels[i] = mapSpeciesToClass(iris_data[i].species);
    }
}


// Test set as a subset of the training set (1/3 balanced of each class <- just for testing)
void createTestSet(double *trainData, double *testData, int *trainLabels, int *testLabels, int numDataSamples){
    for (int i = 0, j = 0; i < numDataSamples; i += 3, j++){
        testData[j*4] = trainData[i*4];
        testData[j*4+1] = trainData[i*4+1];
        testData[j*4+2] = trainData[i*4+2];
        testData[j*4+3] = trainData[i*4+3];
        testLabels[j] = trainLabels[i];
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


void printDataSet(double *trainData, int *trainLabels, int trainSize){
    for(int i = 0; i < trainSize; i++){
        printf("Data[%d]", i);
        for(int j = 0; j < FEATURES; j++){
            int idx = i * FEATURES + j;
            printf("%6.3f", trainData[idx]);
        }
        printf(" -> label: %d\n\n", trainLabels[i]);
    }
}



int main(int argc, char** argv) {

    // Selection of best device
    int device = setBestDevice();
    if (device == -1){
        printf("Kernel launch abort\n");
        return -1;
    }

    int k = 5; // k = 5
    int metric = 1; // Metric distance

    IrisData *iris_data;
    int trainSize;

    // Read the Iris dataset
    if (readIrisDataset("Iris.csv", &iris_data, &trainSize) != 0) {
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
    knnDistances<<< grid, block >>>(d_trainData, d_testData, d_distances, trainSize, testSize, metric);
    cudaDeviceSynchronize();        //forcing synchronous behavior
    double knnDistElaps = cpuSecond() - knnDistStart;
    
    dim3 gridDim(testSize, 1, 1);   // each thread block (1 thread inside) take care of one testData element (trainSize distances to sort)
    dim3 blockDim(1, 1, 1);
    
    double knnSortStart = cpuSecond();
    knnSortPredict<<< gridDim, blockDim >>>(d_distances, trainSize, d_trainIndexes, k, d_predictions, d_trainLabels);
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
    writeResultsToFile(testLabels, predictions, errorCount, testSize, "results.txt", trainSize, FEATURES, k, metric, distDim, predDim, knnDistElaps, knnSortElaps); 
    writeDeviceInfo("device_info.txt", device);

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