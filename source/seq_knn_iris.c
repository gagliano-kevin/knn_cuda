#include "../include/c_functions.h"
#include "../include/iris_functions.h"


int main() {

    printf("Executing file: %s\n\n", __FILE__);

    int k = 10; 
    int metric = 1;                                                                            // Euclidean distance
    int exp = 4;                                                                               // Power for Minkowski distance (not used in this case)

    double avgKnnElaps = 0.0;                                                                  // Average time for kNN execution
    int errorCount = 0;
    int trainSize = 0;
    int testSize = 0;
    double exeTimes[5] = {0.0, 0.0, 0.0, 0.0, 0.0};

    for(int i = 1; i <= 5; i++){                                                                // 5 iterations to calculate average time

        IrisData *iris_data;                                                                    // Pointer to IrisData struct

        // Read the Iris dataset
        if (readIrisDataset("../datasets/Iris.csv", &iris_data, &trainSize) != 0) {
            fprintf(stderr, "Error reading Iris dataset\n");
            return 1;
        }

        testSize = trainSize/3;                                                                 // Test set size is 1/3 of the training set

        // Allocate memory for training set
        double *trainData = (double *)malloc(trainSize * FEATURES * sizeof(double));
        int *trainLabels = (int *)malloc(trainSize * sizeof(int));

        createTrainingSet(iris_data, trainData, trainLabels, trainSize);                        // Training data extraction

        // Test set (1/3 of training set, balanced over classes -> 17,17,16)
        int testDataSize = (trainSize / 3) * FEATURES * sizeof(double);
        int testLabelsSize = (trainSize / 3) * sizeof(int);
        
        // Allocate memory for test set
        double *testData = (double *)malloc(testDataSize);
        int *testLabels = (int *)malloc(testLabelsSize);
        
        createTestSet(trainData, testData, trainLabels, testLabels, trainSize);                 // Test data extraction
        // Allocate memory for distances, trainIndexes and predictions
        double *distances = (double *)malloc(trainSize * testSize * sizeof(double));
        int *trainIndexes = (int *)malloc(trainSize * testSize * sizeof(int));
        int *predictions = (int *)malloc(testSize * sizeof(int));

        createTrainIndexes(trainIndexes, testSize, trainSize);                                  // Create training set indexes for each test set element

        // kNN algorithm execution
        double knnStart = cpuSecond();
        knn(trainData, testData, distances, trainSize, testSize, trainIndexes, k, metric, exp, predictions, trainLabels, FEATURES, CLASSES);
        double knnElaps = cpuSecond() - knnStart;
        exeTimes[i-1] = knnElaps;
        avgKnnElaps += knnElaps;

        //check device results
        errorCount = checkResult(testLabels, predictions, testSize);

        // Free host memory
        free(iris_data);
        free(trainData);
        free(trainLabels);
        free(testData);
        free(testLabels);
        free(distances);
        free(trainIndexes);
        free(predictions);
    }
    avgKnnElaps /= 5;                                                                           // Calculate average execution time
    // Write results to file
    appendRunStatsToFile("iris_c.txt", "iris/", exeTimes, 5);
    appendResultsToFile(errorCount, testSize, "iris_c.txt", "iris/", trainSize, FEATURES, k, metric, exp, avgKnnElaps);
    exeTimeToFile("iris_csv.txt", "iris/", &avgKnnElaps, 1);                                    // Write execution time to csv file

    return 0;
}

