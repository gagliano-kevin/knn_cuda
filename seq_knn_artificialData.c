//gcc -o seq_diabetes seq_knn_diabetes.c -lm

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <sys/sysinfo.h>


#define FEATURES 8
#define CLASSES 2


double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}


// Compute distance between two points based on the selected metric
double computeDistance(double *point1, double *point2, int metric, int exp, int num_features) {
    double distance = 0.0;
    if (metric == 1) { // Euclidean distance
        for (int i = 0; i < num_features; i++) {
            distance += (point1[i] - point2[i]) * (point1[i] - point2[i]);
        }
        distance = sqrt(distance);
    } else if (metric == 2) { // Manhattan distance
        for (int i = 0; i < num_features; i++) {
            distance += fabs(point1[i] - point2[i]);
        }
    } else if (metric == 3) { // Minkowski distance with p = exp
        double sum = 0.0;
        for (int i = 0; i < num_features; i++) {
            sum += pow(fabs(point1[i] - point2[i]), exp);
        }
        distance = pow(sum, 1.0 / (float)exp);
    }
    return distance;
}


// Bubble sort
void bubble_sort(double *distances, int *indexes, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (distances[j] > distances[j + 1]) {
                double temp_dist = distances[j];
                int temp_index = indexes[j];
                distances[j] = distances[j + 1];
                indexes[j] = indexes[j + 1];
                distances[j + 1] = temp_dist;
                indexes[j + 1] = temp_index;
            }
        }
    }
}


// Function to perform KNN classification with bubble sort
// n=trainSize, dim=FEATURES, m=testSize
void knn_bubble(double *trainData, double *testData, double *distances, int trainSize, int testSize, int *indexes, int k, int metric, int exp, int *predictions, int *trainLabels, int num_features, int classes){

    for (int q = 0; q < testSize; q++) {           // for each element in testSet
        for (int i = 0; i < trainSize; i++) {       // for each element in trainSet
            int idx = q * trainSize + i;
            distances[idx] = computeDistance(&trainData[i * num_features], &testData[q * num_features], metric, exp, num_features);
        }
        bubble_sort(&distances[q * trainSize], &indexes[q * trainSize], trainSize);

        // Count occurrences of labels in the k nearest neighbors
        //int classCounts[classes] = {0}; 
        int *classCounts = (int *)malloc(classes * sizeof(int));
        for(int i = 0; i < classes; i++){
            classCounts[i] = 0;
        }
        for (int i = 0; i < k; i++) {
            int index = indexes[q * trainSize + i];
            // Assuming labels are stored in the last column of the data matrix
            classCounts[trainLabels[index]]++;
        }

        // Determine the majority class
        int max_count = 0;
        int predicted_label = -1;
        for (int i = 0; i < classes; i++) {
            if (classCounts[i] > max_count) {
                max_count = classCounts[i];
                predicted_label = i;
            }
        }

        predictions[q] = predicted_label;
        free(classCounts);
    }
}


void writeResultsToFile(int * trainLabels, int *results, int errorCount, int testSize, const char *filename, int trainSize, int features, int k, int metric, int exp, double time1) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }

    fprintf(file, "knn execution information:\n");
    fprintf(file, "knn execution time %f sec\n", time1);

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
        fprintf(file, "Minkowski (p=%d)\n", exp);
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


void write_hardware_specification(const char *filename) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file.\n");
        return;
    }

    // Get system information
    struct sysinfo sys_info;
    if (sysinfo(&sys_info) != 0) {
        printf("Error getting system information.\n");
        fclose(file);
        return;
    }

    // Write hardware specification to file
    fprintf(file, "System Information:\n");
    fprintf(file, "--------------------\n");
    fprintf(file, "Total RAM: %lu MB\n", sys_info.totalram / (1024 * 1024));
    fprintf(file, "Free RAM: %lu MB\n", sys_info.freeram / (1024 * 1024));
    fprintf(file, "Total Swap: %lu MB\n", sys_info.totalswap / (1024 * 1024));
    fprintf(file, "Free Swap: %lu MB\n", sys_info.freeswap / (1024 * 1024));
    fprintf(file, "Number of CPUs: %d\n", sys_info.procs);


    fprintf(file, "\nCPU Information:\n");
    fprintf(file, "----------------\n");

    FILE *cpuinfo = fopen("/proc/cpuinfo", "r");
    if (cpuinfo == NULL) {
        printf("Error opening CPU info file.\n");
        fclose(file);
        return;
    }

    char line[256];
    while (fgets(line, sizeof(line), cpuinfo)) {
        fputs(line, file);
    }

    fclose(cpuinfo);
    fclose(file);

    printf("Hardware specification has been written to %s\n", filename);
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



void printDataSet(double *trainData, int *trainLabels, int trainSize, int num_features){
    for(int i = 0; i < trainSize; i++){
        printf("Data[%d]", i);
        for(int j = 0; j < num_features; j++){
            int idx = i * num_features + j;
            printf("%9.3f", trainData[idx]);
        }
        printf(" -> label: %d\n\n", trainLabels[i]);
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


// Function to generate a Dataset
void generateData(int size, int num_features, double **data, int **labels, double mean) {
    // Allocate memory for data and labels
    *data = (double *)malloc(size * num_features * sizeof(double));
    *labels = (int *)malloc(size * sizeof(int));
    
    // Generate training data
    double noise = 0.1; // Adjust this value to control noise level
    
    srand(time(NULL)); // Seed for random number generation
    
    for (int i = 0; i < size; i++) {
        int class_index = i % num_features;
        
        // Fill data vector with noise
        for (int j = 0; j < num_features; j++) {
            if (j == class_index) {
                // Generate value for class component as sum of mean value and noise
                (*data)[i * num_features + j] = mean + ((double)rand() / RAND_MAX) * noise;
            } else {
                // Other components are noise
                (*data)[i * num_features + j] = ((double)rand() / RAND_MAX) * noise;
            }
        }
        
        // Assign label
        (*labels)[i] = class_index;
    }
}



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

