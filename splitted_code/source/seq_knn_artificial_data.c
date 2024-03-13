#include "../include/c_functions.h"

int main() {

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

    double knnStart = cpuSecond();
    knn_bubble(trainData, testData, distances, trainSize, testSize, trainIndexes, k, metric, exp, predictions, trainLabels, num_features, num_classes);
    double knnElaps = cpuSecond() - knnStart;



    //printDataSet(trainData, trainLabels, trainSize);

    //printDistances(distances, testSize, trainSize);

    //printTrainIndexes(trainIndexes, testSize, trainSize);


    //check device results
    int errorCount = checkResult(testLabels, predictions, testSize);


    // Write results and device info to file
    writeResultsToFile(testLabels, predictions, errorCount, testSize, "seq_results.txt", trainSize, num_features, k, metric, exp, knnElaps);
    const char *filename = "hardware_spec.txt";
    write_hardware_specification(filename);


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

