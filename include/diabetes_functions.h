#ifndef DIABETES_FUNCTIONS_H
#define DIABETES_FUNCTIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_FIELD_LEN 20
#define MAX_FIELDS 9
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
    int label;
} Row;


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
        (*dataset)[i].label = atoi(token);
    }

    fclose(file);
    return 1;
}


// Function to extract the features and labels from the array of structs into separate arrays
void extractData(const Row *dataset, double *features, int *labels, int numRows) {
    for (int i = 0; i < numRows; i++) {
        features[i * FEATURES] = dataset[i].pregnancies;
        features[i * FEATURES + 1] = dataset[i].glucose;
        features[i * FEATURES + 2] = dataset[i].bloodPressure;
        features[i * FEATURES + 3] = dataset[i].skinThickness;
        features[i * FEATURES + 4] = dataset[i].insulin;
        features[i * FEATURES + 5] = dataset[i].bmi;
        features[i * FEATURES + 6] = dataset[i].diabetesPedigreeFunction;
        features[i * FEATURES + 7] = dataset[i].age;

        labels[i] = dataset[i].label;
    }
}



#endif