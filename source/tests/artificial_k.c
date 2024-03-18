#include "../../include/c_functions.h"


int main() {


    printf("Executing file: %s\n\n", __FILE__);

    int k = 5; 
    int metric = 1; // Euclidean distance
    int exp = 4; // Power for Minkowski distance (not used in this case)
    int trainSize = 1000; // Size of the dataset
    int testSize = 100; // Size of the dataset
    int mean = 10; // Mean value for class component
    int num_features = 10; // Number of features (and classes)
    int num_classes = num_features; // Number of classes

    double exeTimes[10];

    for(k = 5; k <= 50; k += 5){


        int errorCount = 0;

        double avgKnnElaps = 0.0;
        for(int i = 1; i <= 10; i++){

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
            knn(trainData, testData, distances, trainSize, testSize, trainIndexes, k, metric, exp, predictions, trainLabels, num_features, num_classes);
            double knnElaps = cpuSecond() - knnStart;
            //printf("Iterations [%d] --> Elapsed time for knn computation: %f\n", i, knnElaps);

            avgKnnElaps += knnElaps;



            //check device results
            errorCount = checkResult(testLabels, predictions, testSize);
            //printf("Error count: %d\n", errorCount);


            // Free host memory
            free(trainData);
            free(trainLabels);
            free(testData);
            free(testLabels);
            free(distances);
            free(trainIndexes);
            free(predictions);
        }
        avgKnnElaps /= 10;
        //printf("Average elapsed time for knn computation: %f\n", avgKnnElaps);

        exeTimes[(k/5)-1] = avgKnnElaps;


        // Print results to file
        appendResultsToFile(errorCount, testSize, "artificial_k_c.txt", "artificial_k/", trainSize, num_features, k, metric, exp, avgKnnElaps);

    }

    exeTimeToFile("artificial_k_csv.txt", "artificial_k/", exeTimes, 10);



    return 0;
}