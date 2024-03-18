#include "../include/c_functions.h"
#include "../include/diabetes_functions.h"

int main() {

    int k = 10; 
    int metric = 1; // Metric distance
    int exp = 4; // Exponent for Minkowski distance

    double avgKnnElaps = 0.0;
    int errorCount = 0;
    int trainSize = 0;
    int testSize = 0;

    printf("Executing file: %s\n\n", __FILE__);

    for(int i = 1; i <= 5; i++){
        Row *dataset;

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


        double knnStart = cpuSecond();
        knn(trainData, testData, distances, trainSize, testSize, trainIndexes, k, metric, exp, predictions, trainLabels, FEATURES, CLASSES);
        double knnElaps = cpuSecond() - knnStart;
        avgKnnElaps += knnElaps;



        //printDataSet(trainData, trainLabels, trainSize);

        //printDistances(distances, testSize, trainSize);

        //printTrainIndexes(trainIndexes, testSize, trainSize);


        //check device results
        errorCount = checkResult(testLabels, predictions, testSize);


        // Write results and device info to file
        //writeResultsToFile(testLabels, predictions, errorCount, testSize, "seq_results_diabetes.txt", "seq_results_diabetes/", trainSize, FEATURES, k, metric, exp, knnElaps);
        //const char *filename = "hardware_spec.txt";
        //writeAllInfoToFile(filename);


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
    avgKnnElaps /= 5;

    appendResultsToFile(errorCount, testSize, "diabetes_c.txt", "diabetes/", trainSize, FEATURES, k, metric, exp, avgKnnElaps);
    exeTimeToFile("diabetes_csv.txt", "diabetes/", &avgKnnElaps, 1);


    return 0;
}

