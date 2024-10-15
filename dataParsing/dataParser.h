#ifndef DATAPARSER_H
#define DATAPARSER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_TEXT_LENGTH 1024

typedef struct {
    char text[MAX_TEXT_LENGTH];
    int label;
} DataPoint;

// Function prototypes
DataPoint* parseCSV(const char* filename, int* num_datapoints);
void freeData(DataPoint* data, int num_datapoints);

#endif