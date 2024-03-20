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


// Get current CPU time in seconds 
double cpuSecond(){
    struct timeval tp;                                              // Structure to store the current time
    gettimeofday(&tp, NULL);                                        // Get the current time and store it in the tp structure
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);          // CPU time in seconds (microseconds part converted to seconds)
}


// Get error count between labels and predictions
int checkResult(int *labels, int *predictions, const int N){
    int errorCount = 0;
    for (int i=0; i<N; i++){
        if(labels[i] != predictions[i]){
            errorCount++;
        }
    }
    return errorCount;
}


// Function to print the dataset
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


// Function to create train indexes (used to access labels in the train set)
void createTrainIndexes(int *trainIndexes, int testSize, int trainSize){
    for(int i = 0; i < testSize; i++){
        for(int j = 0; j < trainSize; j++){
            int index = i * trainSize + j; 
            trainIndexes[index] = j;
        }
    }
}


// Function to print train indexes
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


// Function to print the distances
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


// Function to generate an artificial dataset with a given number of features (clusters of data points around a mean value)
void generateData(int size, int num_features, double **data, int **labels, double mean) {
    // Allocate memory for data and labels
    *data = (double *)malloc(size * num_features * sizeof(double));
    *labels = (int *)malloc(size * sizeof(int));
    
    double noise = 0.1;                                                                         // Value to control noise level
    srand(time(NULL));                                                                          // Seed for random number generation
    
    for (int i = 0; i < size; i++) {
        int class_index = i % num_features;                                                     // Modular arithmetic to assign class index
        // Fill data vector 
        for (int j = 0; j < num_features; j++) {
            if (j == class_index) {
                (*data)[i * num_features + j] = mean + ((double)rand() / RAND_MAX) * noise;     // One component is the mean + noise
            } else {
                (*data)[i * num_features + j] = ((double)rand() / RAND_MAX) * noise;            // Other components are only noise
            }
        }
        (*labels)[i] = class_index;                                                             // Assign label
    }
}


// Function to get compiler information
char* getCompilerInfo() {
    char buffer[BUFFER_SIZE];                                                                   // Buffer to store lines read from the command output
    char* compilerInfo = (char *)malloc(1);                                                     // Allocate memory for the string
    compilerInfo[0] = '\0';                                                                     // Ensure the string is properly terminated
    // Open a pipe to execute the command "gcc --version" 
    FILE* fp = popen("gcc --version", "r");                                                     
    if (fp == NULL) {                                                                           // Check if the pipe was opened successfully
        printf("Failed to run command\n");
        return NULL;
    }
    // Read each line from the command output and append it to the compilerInfo string
    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        // Dynamically reallocate memory to accommodate the new read line in the compilerInfo string
        compilerInfo = (char *)realloc(compilerInfo, strlen(compilerInfo) + strlen(buffer) + 1);
        strcat(compilerInfo, buffer);                                                           // Append the line to the compilerInfo string
    }
    pclose(fp);                                                                                 // Close the pipe
    return compilerInfo; 
}


// Function to get nvcc information
char* getNVCCInfo() {
    char buffer[BUFFER_SIZE];                                                                   // Buffer to store lines read from the command output
    char* nvccInfo = (char *)malloc(1);                                                         // Allocate memory for the string
    nvccInfo[0] = '\0';                                                                         // Ensure the string is properly terminated
    // Open a pipe to execute the command "nvcc --version"
    FILE* fp = popen("nvcc --version", "r");                                                                                              
    if (fp == NULL) {                                                                           // Check if the pipe was opened successfully
        printf("Failed to run command\n");
        return NULL;
    }
    // Read each line from the command output and append it to the nvccInfo string
    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        // Dynamically reallocate memory to accommodate the new read line in the nvccInfo string
        nvccInfo = (char *)realloc(nvccInfo, strlen(nvccInfo) + strlen(buffer) + 1);        
        strcat(nvccInfo, buffer);                                                               // Append the line to the nvccInfo string                   
    }
    pclose(fp);                                                                                 // Close the pipe                
    return nvccInfo;
}


// Function to get OS information
char* getOSInfo() {
    char buffer[BUFFER_SIZE];                                                                   // Buffer to store lines read from the command output
    char* osInfo = (char *)malloc(1);                                                           // Allocate memory for the string     
    osInfo[0] = '\0';                                                                           // Ensure the string is properly terminated
    // Open a pipe to execute the command "uname -a" (all available information about the system)
    FILE* fp = popen("uname -a", "r");                                                          
    if (fp == NULL) {                                                                           // Check if the pipe was opened successfully
        printf("Failed to run command\n");
        return NULL;
    }
    // Read each line from the command output and append it to the osInfo string
    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        // Dynamically reallocate memory to accommodate the new read line in the osInfo string
        osInfo = (char *)realloc(osInfo, strlen(osInfo) + strlen(buffer) + 1);      
        strcat(osInfo, buffer);                                                                 // Append the line to the osInfo string
    }
    pclose(fp);                                                                                 // Close the pipe
    return osInfo;
}


// Check if a directory exists
int directoryExists(const char *path) {
    struct stat info;
    // Get information about the file pointed to by 'path'
    if(stat(path, &info) != 0) {                                                                // Error accessing the directory
        return 0;
    }
    return S_ISDIR(info.st_mode);                                                               // Check if the file mode indicates a directory
}


// Create a directory if it does not exist
int createDirectory(const char* dirname) {
    if(mkdir(dirname, 0777) == 0) {                                                             // Create the directory with full permissions
        //printf("Directory created successfully.\n");
        return 1;                                                                               // Return 1 to indicate success
    } else {
        if(directoryExists(dirname)) {                                                          // Check if the directory already exists
            //printf("Directory already exists.\n");
            return 1;                                                                           // Return 1 to indicate success
        }
        printf("Failed to create directory : %s\n\n", dirname);
        return 0;                                                                               // Return 0 to indicate failure
    }
}


// Function to write execution times to a csv file
void exeTimeToFile(const char *filename, const char *dirname, double* exeTimes, int num_executions){
    createDirectory(dirname); 
    char path[256];                                                                             // Assuming max path length of 256 characters
    snprintf(path, sizeof(path), "%s%s", dirname, filename);                                    // Concatenate dirname and filename
    FILE *file = fopen(path, "a");                                                              // Open file in append mode
    if (file == NULL) { 
        printf("Error opening file!\n");
        return;
    }
    // Write execution times comma separated
    for(int i = 0; i < num_executions; i++){
        fprintf(file, "%f , ", exeTimes[i]);
    }
    fprintf(file, "\n");
    fclose(file);
}


#endif