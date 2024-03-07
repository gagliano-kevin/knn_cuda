//gcc -o seq_iris seq_knn_iris.c -lm

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <sys/sysinfo.h>

#define MAX_LINE_SIZE 1024
#define MAX_FIELD_SIZE 128
#define NUM_FIELDS 6    // Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species
#define FEATURES 4
#define CLASSES 3


typedef struct {
    int id;
    float sepal_length;
    float sepal_width;
    float petal_length;
    float petal_width;
    char species[MAX_FIELD_SIZE];
} IrisData;


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
// n=trainSize(150), dim=FEATURES(4), m=testSize(50)
//knnSortPredict(double *distances, int trainSize, int *indexes, int k, int *predictions, int *trainLabels)
//void knn_bubble(double *data, int n, int dim, double *test_vectors, int m, int k, int **predicted_labels) {
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


int readIrisDataset(const char *filename, IrisData **iris_data, int *num_samples) {
    FILE *file;
    char line[MAX_LINE_SIZE];
    char *token;

    // Open the CSV file
    file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file\n");
        return 1;
    }

    // Skip the first line
    if (fgets(line, sizeof(line), file) == NULL) {
        fprintf(stderr, "Error reading file\n");
        fclose(file);
        return 1;
    }

    int count = 0;
    *num_samples = 0;
    // Count the number of lines (excluding the first)
    while (fgets(line, sizeof(line), file) != NULL) {
        (*num_samples)++;
    }

    // Allocate memory for IrisData array
    *iris_data = (IrisData *)malloc(*num_samples * sizeof(IrisData));
    if (*iris_data == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        return 1;
    }

    // Reset file pointer to the beginning
    fseek(file, 0, SEEK_SET);
    // Skip the first line
    fgets(line, sizeof(line), file);

    // Read each subsequent line of the file
    while (fgets(line, sizeof(line), file) != NULL) {
        IrisData *iris = &((*iris_data)[count]);
        int field_index = 0;

        // Tokenize each line based on the comma delimiter
        token = strtok(line, ",");
        while (token != NULL && field_index < NUM_FIELDS) {
            // Parse and store the tokenized values into appropriate data structures
            switch (field_index) {
                case 0:
                    iris->id = atoi(token);
                    break;
                case 1:
                    iris->sepal_length = atof(token);
                    break;
                case 2:
                    iris->sepal_width = atof(token);
                    break;
                case 3:
                    iris->petal_length = atof(token);
                    break;
                case 4:
                    iris->petal_width = atof(token);
                    break;
                case 5:
                token = strtok(token, "\n");
                    strcpy(iris->species, token);
                    break;
            }
            // Get the next token
            token = strtok(NULL, ",");
            field_index++;
        }
        count++;
    }

    // Close the file
    fclose(file);

    return 0;
}


// Function to map species string to class number
int mapSpeciesToClass(const char *species) {
    if (strcmp(species, "Iris-setosa") == 0) {
        return 0;
    } else if (strcmp(species, "Iris-versicolor") == 0) {
        return 1;
    } else if (strcmp(species, "Iris-virginica") == 0) {
        return 2;
    } else {
        return -1; // Unknown species
    }
}


// Training set composed of all the dataset samples
void createTrainingSet(IrisData *iris_data, double *trainData, int *trainLabels, int numSamples){
    for(int i = 0; i < numSamples; i++){
        trainData[i*4] = iris_data[i].sepal_length;
        trainData[i*4+1] = iris_data[i].sepal_width;
        trainData[i*4+2] = iris_data[i].petal_length;
        trainData[i*4+3] = iris_data[i].petal_width;
        trainLabels[i] = mapSpeciesToClass(iris_data[i].species);
    }
}


// Test set as a subset of the training set (1/3 balanced of each class <- just for testing)
void createTestSet(double *trainData, double *testData, int *trainLabels, int *testLabels, int numDataSamples){
    for (int i = 0, j = 0; i < numDataSamples; i += 3, j++){
        testData[j*4] = trainData[i*4];
        testData[j*4+1] = trainData[i*4+1];
        testData[j*4+2] = trainData[i*4+2];
        testData[j*4+3] = trainData[i*4+3];
        testLabels[j] = trainLabels[i];
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


void printDataSet(double *trainData, int *trainLabels, int trainSize){
    for(int i = 0; i < trainSize; i++){
        printf("Data[%d]", i);
        for(int j = 0; j < FEATURES; j++){
            int idx = i * FEATURES + j;
            printf("%6.3f", trainData[idx]);
        }
        printf(" -> label: %d\n\n", trainLabels[i]);
    }
}



int main() {

    int k = 5; // k = 5
    int metric = 1; // Metric distance

    IrisData *iris_data;
    int trainSize;

    // Read the Iris dataset
    if (readIrisDataset("Iris.csv", &iris_data, &trainSize) != 0) {
        fprintf(stderr, "Error reading Iris dataset\n");
        return 1;
    }

    int testSize = trainSize/3;

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
    free(iris_data);
    free(trainData);
    free(trainLabels);
    free(testData);
    free(testLabels);
    free(distances);
    free(trainIndexes);
    free(predictions);


    return 0;
}

