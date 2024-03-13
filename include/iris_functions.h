#ifndef DIABETES_FUNCTIONS_H
#define DIABETES_FUNCTIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_SIZE 1024
#define MAX_FIELD_SIZE 128
#define NUM_FIELDS 6  
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
        trainData[i*FEATURES] = iris_data[i].sepal_length;
        trainData[i*FEATURES+1] = iris_data[i].sepal_width;
        trainData[i*FEATURES+2] = iris_data[i].petal_length;
        trainData[i*FEATURES+3] = iris_data[i].petal_width;
        trainLabels[i] = mapSpeciesToClass(iris_data[i].species);
    }
}


// Test set as a subset of the training set (1/3 balanced of each class <- just for testing)
void createTestSet(double *trainData, double *testData, int *trainLabels, int *testLabels, int numDataSamples){
    for (int i = 0, j = 0; i < numDataSamples; i += 3, j++){
        testData[j*FEATURES] = trainData[i*FEATURES];
        testData[j*FEATURES+1] = trainData[i*FEATURES+1];
        testData[j*FEATURES+2] = trainData[i*FEATURES+2];
        testData[j*FEATURES+3] = trainData[i*FEATURES+3];
        testLabels[j] = trainLabels[i];
    }
}




#endif