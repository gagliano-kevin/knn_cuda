#include "../include/c_functions.h"
#include "../include/iris_functions.h"


int main() {

    int k = 5; 
    int metric = 1; // Metric distance
    int exp = 4; // Minkowski exponent
    double avgKnnElaps = 0.0;
    int errorCount = 0;
    int trainSize = 0;
    int testSize = 0;

    printf("Executing file: %s\n\n", __FILE__);

    for(int i = 1; i <= 5; i++){
        IrisData *iris_data;

        // Read the Iris dataset
        if (readIrisDataset("../datasets/Iris.csv", &iris_data, &trainSize) != 0) {
            fprintf(stderr, "Error reading Iris dataset\n");
            return 1;
        }

        testSize = trainSize/3;

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
        //writeResultsToFile(testLabels, predictions, errorCount, testSize, "seq_results_iris.txt", "seq_results_iris/", trainSize, FEATURES, k, metric, exp, knnElaps);
        //const char *filename = "hardware_spec.txt";
        //writeAllInfoToFile(filename);


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
    avgKnnElaps /= 5;

    appendResultsToFile(errorCount, testSize, "iris_c.txt", "iris/", trainSize, FEATURES, k, metric, exp, avgKnnElaps);
    exeTimeToFile("iris_csv.txt", "iris/", &avgKnnElaps, 1);


    return 0;
}

