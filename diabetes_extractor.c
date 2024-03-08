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
void extractFeaturesAndOutcomes(const Row *dataset, double **features, int *outcomes, int numRows) {
    for (int i = 0; i < numRows; i++) {
        features[i][0] = dataset[i].pregnancies;
        features[i][1] = dataset[i].glucose;
        features[i][2] = dataset[i].bloodPressure;
        features[i][3] = dataset[i].skinThickness;
        features[i][4] = dataset[i].insulin;
        features[i][5] = dataset[i].bmi;
        features[i][6] = dataset[i].diabetesPedigreeFunction;
        features[i][7] = dataset[i].age;

        outcomes[i] = dataset[i].outcome;
    }
}

int main() {
    Row *dataset;
    int numRows;

    if (readCSV("diabetes_training.csv", &dataset, &numRows) != 1) {
        printf("Error reading CSV file.\n");
        return 1;
    }

    // Allocate memory for trainData
    double **trainData = (double **)malloc(numRows * sizeof(double *));
    if (trainData == NULL) {
        printf("Error allocating memory.\n");
        free(dataset);
        return 1;
    }
    for (int i = 0; i < numRows; i++) {
        trainData[i] = (double *)malloc(FEATURES * sizeof(double));
        if (trainData[i] == NULL) {
            printf("Error allocating memory.\n");
            for (int j = 0; j < i; j++) {
                free(trainData[j]);
            }
            free(trainData);
            free(dataset);
            return 1;
        }
    }

    // Allocate memory for labels
    int *labels = malloc(numRows * sizeof(int));
    if (labels == NULL) {
        printf("Error allocating memory.\n");
        for (int i = 0; i < numRows; i++) {
            free(trainData[i]);
        }
        free(trainData);
        free(dataset);
        return 1;
    }

    extractFeaturesAndOutcomes(dataset, trainData, labels, numRows);

    // Output the extracted data
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < FEATURES; j++) {
            printf("%f ", trainData[i][j]);
        }
        printf("%d\n", labels[i]);
    }

    // Free allocated memory
    for (int i = 0; i < numRows; i++) {
        free(trainData[i]);
    }
    free(trainData);
    free(labels);
    free(dataset);

    return 0;
}