#include "../include/c_functions.h"
#include "../include/diabetes_functions.h"

int main() {

    printf("Executing file: %s\n\n", __FILE__);

    int k = 10; 
    int metric = 1;                                                                            // Euclidean distance
    int exp = 4;                                                                               // Power for Minkowski distance (not used in this case)

    double avgKnnElaps = 0.0;                                                                  // Average time for kNN execution
    int errorCount = 0;
    int trainSize = 0;
    int testSize = 0;

    for(int i = 1; i <= 5; i++){                                                                // 5 iterations to calculate average time

        Row *dataset;                                                                           // Pointer to Row struct                         

        // Training data
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
        
        // Test data
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

        // Host memory allocation
        double *distances = (double *)malloc(trainSize * testSize * sizeof(double));
        int *trainIndexes = (int *)malloc(trainSize * testSize * sizeof(int));
        int *predictions = (int *)malloc(testSize * sizeof(int));

        createTrainIndexes(trainIndexes, testSize, trainSize);                                      // Create training set indexes for each test set element

        // Knn execution
        double knnStart = cpuSecond();
        knn(trainData, testData, distances, trainSize, testSize, trainIndexes, k, metric, exp, predictions, trainLabels, FEATURES, CLASSES);
        double knnElaps = cpuSecond() - knnStart;
        avgKnnElaps += knnElaps;

        //check device results
        errorCount = checkResult(testLabels, predictions, testSize);

        // Free host memory
        free(dataset);
        free(trainData);
        free(trainLabels);
        free(testData);
        free(testLabels);
        free(distances);
        free(trainIndexes);
        free(predictions);
    }
    avgKnnElaps /= 5;                                                                               // Calculate average execution time
    // Write results file
    appendResultsToFile(errorCount, testSize, "diabetes_c.txt", "diabetes/", trainSize, FEATURES, k, metric, exp, avgKnnElaps); 
    exeTimeToFile("diabetes_csv.txt", "diabetes/", &avgKnnElaps, 1);                                // Write execution time to csv file

    return 0;
}

