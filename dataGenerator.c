#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Function to generate a training set
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

// Example usage
int main() {
    int size = 100; // Size of the dataset
    int num_features = 5; // Number of features (and classes)
    int mean = 10; // Mean value for class component
    
    // Allocate memory for data and labels
    double *data;
    int *labels;
    
    // Generate training set
    generateData(size, num_features, &data, &labels, mean);
    
    // Print generated data and labels
    printf("Generated Data:\n");
    for (int i = 0; i < size; i++) {
        printf("Data Point %d: ", i);
        for (int j = 0; j < num_features; j++) {
            printf("%.2f ", data[i * num_features + j]);
        }
        printf("Label: %d\n", labels[i]);
    }
    
    // Free allocated memory
    free(data);
    free(labels);
    
    return 0;
}
