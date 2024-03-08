#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define MAX_FIELD_LEN 20
#define MAX_FIELDS 9
#define FEATURES 8

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

int main() {
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
    int *trainLabels = malloc(trainSize * sizeof(int));
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
    int *testLabels = malloc(testSize * sizeof(int));
    if (testLabels == NULL) {
        printf("Error allocating memory.\n");
        free(dataset);
        free(testData);
        return 1;
    }

    // Test data extraction
    extractFeaturesAndOutcomes(dataset, testData, testLabels, testSize);
    

    //printDataSet(testData, testLabels, testSize);
    

    // Free allocated memory
    free(testData);
    free(testLabels);
    free(trainData);
    free(trainLabels);
    free(dataset);

    return 0;
}