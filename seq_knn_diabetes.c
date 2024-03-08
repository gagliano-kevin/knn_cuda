//gcc -o seq_diabetes seq_knn_diabetes.c -lm

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <sys/sysinfo.h>

#define MAX_FIELD_LEN 20
#define MAX_FIELDS 9
#define NUM_FIELDS 6    // Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species
#define FEATURES 8
#define CLASSES 2


// Define a struct to represent each row of the dataset
typedef struct {
    double pregnancies;
    double glucose;
    double bloodPressure;
    double skinThickness;
    double insulin;
    double bmi;
    double diabetesPedigreeFunction;
    double age;
    int outcome;
} Row;


double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}


// Compute distance between two points based on the selected metric
double computeDistance(double *point1, double *point2, int metric) {
    double distance = 0.0;
    if (metric == 1) { // Euclidean distance
        for (int i = 0; i < FEATURES; i++) {
            distance += (point1[i] - point2[i]) * (point1[i] - point2[i]);
            //distance += pow((point1[i] - point2[i]),2);
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


// Function to calculate Euclidean distance between two vectors
double distance(double *v1, double *v2, int dim) {
    double sum = 0;
    for (int i = 0; i < dim; i++) {
        sum += pow(v2[i] - v1[i], 2);
    }
    return sqrt(sum);
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
void knn_bubble(double *trainData, double *testData, double *distances, int trainSize, int testSize, int *indexes, int k, int metric, int *predictions, int *trainLabels){

    for (int q = 0; q < testSize; q++) {           // for each element in testSet
        for (int i = 0; i < trainSize; i++) {       // for each element in trainSet
            int idx = q * trainSize + i;
            distances[idx] = computeDistance(&trainData[i * FEATURES], &testData[q * FEATURES], metric);
        }
        bubble_sort(&distances[q * trainSize], &indexes[q * trainSize], trainSize);

        // Count occurrences of labels in the k nearest neighbors
        int classCounts[CLASSES] = {0}; 
        for (int i = 0; i < k; i++) {
            int index = indexes[q * trainSize + i];
            // Assuming labels are stored in the last column of the data matrix
            classCounts[trainLabels[index]]++;
        }

        // Determine the majority class
        int max_count = 0;
        int predicted_label = -1;
        for (int i = 0; i < CLASSES; i++) {
            if (classCounts[i] > max_count) {
                max_count = classCounts[i];
                predicted_label = i;
            }
        }

        predictions[q] = predicted_label;
    }
}


void writeResultsToFile(int * trainLabels, int *results, int errorCount, int testSize, const char *filename, int trainSize, int features, int k, int metric, double time1) {
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


int readCSV(const char *filename, Row **dataset, int *numRows) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file.\n");
        return 0;
    }

    char line[MAX_FIELDS * MAX_FIELD_LEN]; // Adjusted size for line buffer
    char *token;

    // Skip the first line (column headers)
    if (fgets(line, sizeof(line), file) == NULL) {
        printf("Error reading file.\n");
        fclose(file);
        return 0;
    }

    *numRows = 0;
    // Calculate the total number of rows in the file
    while (fgets(line, sizeof(line), file) != NULL) {
        (*numRows)++;
    }

    // Allocate memory for the dataset
    *dataset = (Row *)malloc(*numRows * sizeof(Row));
    if (*dataset == NULL) {
        printf("Error allocating memory.\n");
        fclose(file);
        return 0;
    }

    // Reset file pointer to beginning of the file
    fseek(file, 0, SEEK_SET);

    // Skip the first line (column headers)
    fgets(line, sizeof(line), file);

    // Read data into the dataset
    for (int i = 0; i < *numRows; i++) {
        if (fgets(line, sizeof(line), file) == NULL) {
            printf("Error reading file.\n");
            fclose(file);
            free(*dataset); // Free allocated memory before returning
            return 0;
        }
        token = strtok(line, ",");
        (*dataset)[i].pregnancies = atof(token);

        token = strtok(NULL, ",");
        (*dataset)[i].glucose = atof(token);

        token = strtok(NULL, ",");
        (*dataset)[i].bloodPressure = atof(token);

        token = strtok(NULL, ",");
        (*dataset)[i].skinThickness = atof(token);

        token = strtok(NULL, ",");
        (*dataset)[i].insulin = atof(token);

        token = strtok(NULL, ",");
        (*dataset)[i].bmi = atof(token);

        token = strtok(NULL, ",");
        (*dataset)[i].diabetesPedigreeFunction = atof(token);

        token = strtok(NULL, ",");
        (*dataset)[i].age = atof(token);

        token = strtok(NULL, ",");
        (*dataset)[i].outcome = atoi(token);
    }

    fclose(file);
    return 1;
}


// Function to extract the features and outcomes from the array of structs into separate arrays
void extractFeaturesAndOutcomes(const Row *dataset, double *features, int *outcomes, int numRows) {
    for (int i = 0; i < numRows; i++) {
        features[i * FEATURES] = dataset[i].pregnancies;
        features[i * FEATURES + 1] = dataset[i].glucose;
        features[i * FEATURES + 2] = dataset[i].bloodPressure;
        features[i * FEATURES + 3] = dataset[i].skinThickness;
        features[i * FEATURES + 4] = dataset[i].insulin;
        features[i * FEATURES + 5] = dataset[i].bmi;
        features[i * FEATURES + 6] = dataset[i].diabetesPedigreeFunction;
        features[i * FEATURES + 7] = dataset[i].age;

        outcomes[i] = dataset[i].outcome;
    }
}


void printDataSet(double *trainData, int *trainLabels, int trainSize){
    for(int i = 0; i < trainSize; i++){
        printf("Data[%d]", i);
        for(int j = 0; j < FEATURES; j++){
            int idx = i * FEATURES + j;
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






int main() {

    int k = 10; 
    int metric = 1; // Metric distance

    Row *dataset;
    int trainSize;
    int testSize;

    // TRAINING DATA
    if (readCSV("diabetes_training.csv", &dataset, &trainSize) != 1) {
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
    extractFeaturesAndOutcomes(dataset, trainData, trainLabels, trainSize);
    //printDataSet(trainData, trainLabels, numRows);

    
    // TEST DATA
    if (readCSV("diabetes_testing.csv", &dataset, &testSize) != 1) {
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
    extractFeaturesAndOutcomes(dataset, testData, testLabels, testSize);
    //printDataSet(testData, testLabels, testSize);


    double *distances = (double *)malloc(trainSize * testSize * sizeof(double));
    int *trainIndexes = (int *)malloc(trainSize * testSize * sizeof(int));
    int *predictions = (int *)malloc(testSize * sizeof(int));


    createTrainIndexes(trainIndexes, testSize, trainSize);

    // Bitonic sort <----------------------------------------------------------- knn_bitonic assumes that labels are in the last column of trainData (must be modify)
    //int *predicted_labels_bitonic;
    //knn_bitonic();


    double knnStart = cpuSecond();
    knn_bubble(trainData, testData, distances, trainSize, testSize, trainIndexes, k, metric, predictions, trainLabels);
    double knnElaps = cpuSecond() - knnStart;



    //printDataSet(trainData, trainLabels, trainSize);

    //printDistances(distances, testSize, trainSize);

    //printTrainIndexes(trainIndexes, testSize, trainSize);


    //check device results
    int errorCount = checkResult(testLabels, predictions, testSize);


    // Write results and device info to file
    writeResultsToFile(testLabels, predictions, errorCount, testSize, "seq_results.txt", trainSize, FEATURES, k, metric, knnElaps);
    const char *filename = "hardware_spec.txt";
    write_hardware_specification(filename);


    // Free host memory
    free(dataset);
    free(trainData);
    free(trainLabels);
    free(testData);
    free(testLabels);
    free(distances);
    free(trainIndexes);
    free(predictions);


    return 0;
}

