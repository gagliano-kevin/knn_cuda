#include "../include/c_functions.h"

int main() {

    printf("Executing file: %s\n\n", __FILE__);

    int k = 10; 
    int metric = 1;                                                                                          // Euclidean distance
    int exp = 4;                                                                                             // Power for Minkowski distance (not used in this case)
    int trainSize = 1000;                                                                                    // Size of the training set
    int testSize = 100;                                                                                      // Size of the test set
    int mean = 10;                                                                                           // Mean value for data generation
    int num_features = 10;                                                                                   // Starting number of features (and classes)
    int num_classes = num_features;                                                                          // Number of classes
    
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

    double knnStart = cpuSecond();
    knn(trainData, testData, distances, trainSize, testSize, trainIndexes, k, metric, exp, predictions, trainLabels, num_features, num_classes);
    double knnElaps = cpuSecond() - knnStart;

    //check device results
    int errorCount = checkResult(testLabels, predictions, testSize);

    // Write results and device info to file
    writeResultsToFile(testLabels, predictions, errorCount, testSize, "seq_results_artificial.txt", "seq_results_artificial/", trainSize, num_features, k, metric, exp, knnElaps);
    const char *filename = "hardware_spec.txt";
    writeAllInfoToFile(filename);

    // Free host memory
    free(trainData);
    free(trainLabels);
    free(testData);
    free(testLabels);
    free(distances);
    free(trainIndexes);
    free(predictions);

    return 0;
}

