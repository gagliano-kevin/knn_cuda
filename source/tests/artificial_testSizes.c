#include "../../include/c_functions.h"


int main() {

    printf("Executing file: %s\n\n", __FILE__);

    int k = 10; 
    int metric = 1;                                                                                             // Euclidean distance
    int exp = 4;                                                                                                // Power for Minkowski distance (not used in this case)
    int trainSize = 1000;                                                                                       // Size of the training set
    int testSize = 100;                                                                                         // Size of the test set
    int mean = 10;                                                                                              // Mean value for data generation
    int num_features = 10;                                                                                      // Number of features (and classes)
    int num_classes = num_features;                                                                             // Number of classes
    double exeTimes[10];                                                                                        // Execution times for each test set size 

    // Loop over different test set sizes
    for(testSize = 100; testSize <= 1000; testSize += 100){

        int errorCount = 0;
        double avgKnnElaps = 0.0;

        for(int i = 1; i <= 5; i++){                                                                            // 5 iterations for average time

            // Pointers to memory for data and labels
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

            avgKnnElaps += knnElaps;

            errorCount = checkResult(testLabels, predictions, testSize);                                             // Check the number of errors in the predictions

            // Free host memory
            free(trainData);
            free(trainLabels);
            free(testData);
            free(testLabels);
            free(distances);
            free(trainIndexes);
            free(predictions);
        }
        avgKnnElaps /= 5;

        exeTimes[(testSize/100)-1] = avgKnnElaps;                                                                   // Store the execution time for the current test set size

        // Print results to file
        appendResultsToFile(errorCount, testSize, "artificial_testSizes_c.txt", "artificial_testSizes/", trainSize, num_features, k, metric, exp, avgKnnElaps);
    }

    exeTimeToFile("artificial_testSizes_csv.txt", "artificial_testSizes/", exeTimes, 10);

    return 0;
}