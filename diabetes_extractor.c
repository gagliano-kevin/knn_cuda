#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ROWS 1000
#define MAX_FIELD_LEN 20

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

// Function to read the CSV file and store the data into an array of structs
int readCSV(const char *filename, Row dataset[], int *numRows) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file.\n");
        return 0;
    }

    char line[MAX_ROWS];
    char *token;

    // Skip the first line (column headers)
    if (fgets(line, sizeof(line), file) == NULL) {
        printf("Error reading file.\n");
        fclose(file);
        return 0;
    }

    *numRows = 0;
    while (fgets(line, sizeof(line), file) != NULL && *numRows < MAX_ROWS) {
        token = strtok(line, ",");
        dataset[*numRows].pregnancies = atof(token);

        token = strtok(NULL, ",");
        dataset[*numRows].glucose = atof(token);

        token = strtok(NULL, ",");
        dataset[*numRows].bloodPressure = atof(token);

        token = strtok(NULL, ",");
        dataset[*numRows].skinThickness = atof(token);

        token = strtok(NULL, ",");
        dataset[*numRows].insulin = atof(token);

        token = strtok(NULL, ",");
        dataset[*numRows].bmi = atof(token);

        token = strtok(NULL, ",");
        dataset[*numRows].diabetesPedigreeFunction = atof(token);

        token = strtok(NULL, ",");
        dataset[*numRows].age = atof(token);

        token = strtok(NULL, ",");
        dataset[*numRows].outcome = atoi(token);

        (*numRows)++;
    }

    fclose(file);
    return 1;
}

// Function to extract the features and outcomes from the array of structs into separate arrays
void extractFeaturesAndOutcomes(const Row dataset[], double features[][8], int outcomes[], int numRows) {
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
    Row dataset[MAX_ROWS];
    double features[MAX_ROWS][8];
    int outcomes[MAX_ROWS];
    int numRows;

    if (readCSV("diabetes_training.csv", dataset, &numRows)) {
        extractFeaturesAndOutcomes(dataset, features, outcomes, numRows);

        // Now you can use the 'features' and 'outcomes' arrays for further processing
    }

    for(int i = 0; i < numRows; i++){
        for(int j = 0; j < 8; j++){
            printf("%f ", features[i][j]);
        }
        printf("%d\n", outcomes[i]);
    }   

    return 0;
}
