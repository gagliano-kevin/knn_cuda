#ifndef COMMON_H
#define COMMON_H


#include <sys/time.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>


#define BUFFER_SIZE 256


double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
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

int directoryExists(const char *path) {
    struct stat info;
    if(stat(path, &info) != 0) {
        // Error accessing the directory
        return 0;
    }
    return S_ISDIR(info.st_mode);
}

int createDirectory(const char* dirname) {
    // Attempt to create the directory
    if(mkdir(dirname, 0777) == 0) {
        //printf("Directory created successfully.\n");
        return 1; // Return 1 to indicate success
    } else {
        if(directoryExists(dirname)) {
            //printf("Directory already exists.\n");
            return 1; // Return 1 to indicate success
        }
        printf("Failed to create directory : %s\n\n", dirname);
        return 0; // Return 0 to indicate failure
    }
}


#endif