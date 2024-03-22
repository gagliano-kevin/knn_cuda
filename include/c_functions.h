#ifndef C_FUNCTIONS_H
#define C_FUNCTIONS_H

#include "../include/common.h"
#include <math.h>
#include <sys/sysinfo.h>


// Compute distance between two points based on the selected metric
double computeDistance(double *point1, double *point2, int metric, int exp, int num_features) {
    double distance = 0.0;
    if (metric == 1) {                                                      // Euclidean distance
        for (int i = 0; i < num_features; i++) {
            distance += (point1[i] - point2[i]) * (point1[i] - point2[i]);
        }
        distance = sqrt(distance);
    } else if (metric == 2) {                                               // Manhattan distance
        for (int i = 0; i < num_features; i++) {
            distance += fabs(point1[i] - point2[i]);
        }
    } else if (metric == 3) {                                               // Minkowski distance 
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


// KNN classification algorithm with bubble sort (for distances sorting)
void knn(double *trainData, double *testData, double *distances, int trainSize, int testSize, int *indexes, int k, int metric, int exp, int *predictions, int *trainLabels, int num_features, int classes){
    for (int q = 0; q < testSize; q++) {                                                                                                // for each element in testSet
        for (int i = 0; i < trainSize; i++) {                                                                                           // for each element in trainSet
            int idx = q * trainSize + i;                                                                                                // index of the distance between test element q and train element i
            distances[idx] = computeDistance(&trainData[i * num_features], &testData[q * num_features], metric, exp, num_features);     // compute distance between test element q and train element i
        }
        bubble_sort(&distances[q * trainSize], &indexes[q * trainSize], trainSize);                                                     // sort the distances and relative indexes

        int *classCounts = (int *)malloc(classes * sizeof(int));                                                                        // allocate memory for classCounts (array for counting occurrences of each class in the k nearest neighbors)
        for(int i = 0; i < classes; i++){                                                                                               // initialize classCounts elements to 0           
            classCounts[i] = 0;
        }
        // Count of the occurrences of each class in the k nearest neighbors
        for (int i = 0; i < k; i++) {                   
            int index = indexes[q * trainSize + i];                                                                                     // index of the i-th nearest neighbor
            classCounts[trainLabels[index]]++;                                                                                          // increment the count of the class of the i-th nearest neighbor
        }   
        // Determine the majority class among the k nearest neighbors
        int max_count = 0;                                                                                                              
        int predicted_label = -1;               
        for (int i = 0; i < classes; i++) { 
            if (classCounts[i] > max_count) {
                max_count = classCounts[i];
                predicted_label = i;
            }
        }
        predictions[q] = predicted_label;                                                                                               // assign the predicted label to the q-th test element (outer loop index)
        free(classCounts);                                                                                                              
    }
}


// Append execution information and results to file 
void appendResultsToFile(int errorCount, int testSize, const char *filename, const char *dirname, int trainSize, int features, int k, int metric, int exp, double exeTime){
    createDirectory(dirname); 
    char path[256];                                                                                        // Assuming max path length of 256 characters
    snprintf(path, sizeof(path), "%s%s", dirname, filename);                                               // Concatenate dirname and filename to create the full path
    FILE *file = fopen(path, "a");                                                                         // Open file in append mode
    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }
    fprintf(file, "knn execution information:\n");
    fprintf(file, "knn execution time: %f sec\n", exeTime);
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
    fprintf(file, "\n-------------------------------------------------------------------------------------------------------------------------------------------\n\n");

    fclose(file);
}


// Write execution information and detailed classification results to file
void writeResultsToFile(int * trainLabels, int *results, int errorCount, int testSize, const char *filename, const char *dirname, int trainSize, int features, int k, int metric, int exp, double exeTime) {
    createDirectory(dirname); 
    char path[256];                                                                                         // Assuming max path length of 256 characters
    snprintf(path, sizeof(path), "%s%s", dirname, filename);                                                // Concatenate dirname and filename to create the full path
    FILE *file = fopen(path, "w");                                                                          // Open file in write mode
    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }
    fprintf(file, "knn execution information:\n");
    fprintf(file, "knn execution time: %f sec\n", exeTime);
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
    // Classification results
    fprintf(file, "\nNumber of prediction errors: %d\n", errorCount);
    fprintf(file, "\nResults:\n");
    char outcome[25];                                                                                       // String to store the outcome of the classification
    for (int i = 0; i < testSize; ++i) {    
        if(results[i] == trainLabels[i]){
            strcpy(outcome, "correctly classified");
        } else {
            strcpy(outcome, "incorrectly classified");
        }
        fprintf(file, "Test example %3d  %-22s -> Predicted class: %1d , Expected class: %1d\n", i + 1, outcome, results[i], trainLabels[i]);
    }
    fclose(file);
    printf("Execution results has been written to %s\n\n", path);   
}


// Write hardware and software information of the running system to file
void writeAllInfoToFile(const char *filename) {
    const char* dirname = "sw_hw_info/"; 
    createDirectory(dirname); 
    char path[256];                                                                                         // Assuming max path length of 256 characters
    snprintf(path, sizeof(path), "%s%s", dirname, filename);                                                // Concatenate dirname and filename to create the full path
    FILE *file = fopen(path, "w");
    if (file == NULL) {
        printf("Error opening file.\n");
        return;
    }
    // Get compiler information
    fprintf(file, "Compiler information:\n");
    char* compilerInfo = getCompilerInfo();
    if (compilerInfo != NULL) {
        fprintf(file, "%s\n", compilerInfo);
        free(compilerInfo); 
    }
    // Get operating system information
    fprintf(file, "Operating System information:\n");
    char* osInfo = getOSInfo();
    if (osInfo != NULL) {
        fprintf(file, "%s\n", osInfo);
        free(osInfo); 
    }
    fprintf(file, "\n\n");
    // Get system information
    struct sysinfo sys_info;
    if (sysinfo(&sys_info) != 0) {
        printf("Error getting system information.\n");
        fclose(file);
        return;
    }
    // Write system memory informations and number of processes to file
    fprintf(file, "System Information:\n");
    fprintf(file, "--------------------\n");
    fprintf(file, "Total RAM: %lu MB\n", sys_info.totalram / (1024 * 1024));
    fprintf(file, "Free RAM: %lu MB\n", sys_info.freeram / (1024 * 1024));
    fprintf(file, "Total Swap: %lu MB\n", sys_info.totalswap / (1024 * 1024));
    fprintf(file, "Free Swap: %lu MB\n", sys_info.freeswap / (1024 * 1024));
    fprintf(file, "Number of procs: %d\n", sys_info.procs);
    // Write CPU information to file
    fprintf(file, "\nCPU Information:\n");
    fprintf(file, "----------------\n");
    FILE *cpuinfo = fopen("/proc/cpuinfo", "r");                                                           // Open CPU info file
    if (cpuinfo == NULL) {
        printf("Error opening CPU info file.\n");
        fclose(file);
        return;
    }
    char line[256];                                                                                        // Assuming max line length of 256 characters
    while (fgets(line, sizeof(line), cpuinfo)) {
        fputs(line, file);
    }
    fclose(cpuinfo);
    fclose(file);
    printf("Hardware and Software specification has been written to %s\n\n", path);
}


#endif