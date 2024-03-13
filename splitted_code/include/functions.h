// functions.h
#ifndef FUNCTIONS_H
#define FUNCTIONS_H


double cpuSecond();

void writeResultsToFile(int *trainLabels, int *results, int errorCount, int testSize, const char *filename, int trainSize, int features, int k, int metric, int exp, unsigned int *distDim, unsigned int *predDim, int workers, int alpha, int beta, double kernelTime1, double kernelTime2);

int checkResult(int *labels, int *predictions, const int N);

void printDataSet(double *trainData, int *trainLabels, int trainSize, int num_features);

void createTrainIndexes(int *trainIndexes, int testSize, int trainSize);

void printTrainIndexes(int *trainIndexes, int testSize, int trainSize);

void printDistances(double *distances, int testSize, int trainSize);

void generateData(int size, int num_features, double **data, int **labels, double mean);

char* getCompilerInfo();

char* getNVCCInfo();

char* getOSInfo();







#endif // FUNCTIONS_H
