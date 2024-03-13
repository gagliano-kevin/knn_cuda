// functions.c
#include "../include/functions.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <sys/sysinfo.h>

#define FILE_NAME __FILE__
#define LINE __LINE__
#define BUFFER_SIZE 256



double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}


void writeResultsToFile(int *trainLabels, int *results, int errorCount, int testSize, const char *filename, int trainSize, int features, int k, int metric, int exp, unsigned int *distDim, unsigned int *predDim, int workers, int alpha, int beta, double kernelTime1, double kernelTime2) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }

    fprintf(file, "Kernel launch information:\n");
    fprintf(file, "Grid dimension in knnDistances kernel: %u , %u\n", distDim[0], distDim[1]);
    fprintf(file, "Block dimension in knnDistances kernel: %u , %u\n", distDim[2], distDim[3]);
    fprintf(file, "Grid dimension in knnSortPredict kernel: %u , %u\n",predDim[0], predDim[1]);
    fprintf(file, "Block dimension in knnSortPredict kernel: %u , %u\n", predDim[2], predDim[3]);
    fprintf(file, "Number of workers: %d\n", workers);
    fprintf(file, "Factor alpha: %d\n", alpha);
    fprintf(file, "Factor beta: %d\n", beta);
    fprintf(file, "knnDistances execution time %f sec\n", kernelTime1);
    fprintf(file, "knnSortPredict execution time %f sec\n", kernelTime2);

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


char* getCompilerInfo() {
    char buffer[BUFFER_SIZE];
    char* compilerInfo = (char *)malloc(1); // Allocate memory for the string
    compilerInfo[0] = '\0'; // Ensure the string is properly terminated

    FILE* fp = popen("gcc --version", "r");
    if (fp == NULL) {
        printf("Failed to run command\n");
        return NULL;
    }

    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        // Append the line to the compilerInfo string
        compilerInfo = (char *)realloc(compilerInfo, strlen(compilerInfo) + strlen(buffer) + 1);
        strcat(compilerInfo, buffer);
    }

    pclose(fp);

    return compilerInfo;
}


char* getNVCCInfo() {
    char buffer[BUFFER_SIZE];
    char* nvccInfo = (char *)malloc(1); // Allocate memory for the string
    nvccInfo[0] = '\0'; // Ensure the string is properly terminated

    FILE* fp = popen("nvcc --version", "r");
    if (fp == NULL) {
        printf("Failed to run command\n");
        return NULL;
    }

    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        // Append the line to the nvccInfo string
        nvccInfo = (char *)realloc(nvccInfo, strlen(nvccInfo) + strlen(buffer) + 1);        //
        strcat(nvccInfo, buffer);
    }

    pclose(fp);

    return nvccInfo;
}


char* getOSInfo() {
    char buffer[BUFFER_SIZE];
    char* osInfo = (char *)malloc(1); // Allocate memory for the string
    osInfo[0] = '\0'; // Ensure the string is properly terminated

    FILE* fp = popen("uname -a", "r");
    if (fp == NULL) {
        printf("Failed to run command\n");
        return NULL;
    }

    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        osInfo = (char *)realloc(osInfo, strlen(osInfo) + strlen(buffer) + 1);      // Reallocate memory for the string
        strcat(osInfo, buffer);    // Append the line to the osInfo string
    }

    pclose(fp);

    return osInfo;
}