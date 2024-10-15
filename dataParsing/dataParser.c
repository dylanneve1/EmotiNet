// parseCSV.c
#include "dataParser.h"

DataPoint* parseCSV(const char* filename, int* num_datapoints) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        perror("Error opening file in parseCSV");
        return NULL;
    }

    *num_datapoints = 0;
    char line[MAX_TEXT_LENGTH * 4]; // Increased buffer size to handle longer lines
    DataPoint* data = NULL;
    int capacity = 10000; // Initial capacity

    data = (DataPoint*)malloc(capacity * sizeof(DataPoint));
    if (!data) {
        perror("Memory allocation failed in parseCSV");
        fclose(fp);
        return NULL;
    }

    // Read and skip header line
    if (!fgets(line, sizeof(line), fp)) {
        fprintf(stderr, "CSV file is empty or unreadable.\n");
        free(data);
        fclose(fp);
        return NULL;
    }

    int line_number = 1; // Start counting from the first data line

    while (fgets(line, sizeof(line), fp)) {
        line_number++;

        // Remove potential newline character
        line[strcspn(line, "\n")] = '\0';

        // Find the position of the last comma
        char* last_comma = strrchr(line, ',');
        if (!last_comma) {
            fprintf(stderr, "Invalid data format on line %d: Missing comma. Skipping.\n", line_number);
            continue; // Skip this data point
        }

        // Extract label
        char* label_str = last_comma + 1;
        // Trim whitespace from label_str
        while (isspace((unsigned char)*label_str)) label_str++;
        if (*label_str == '\0') {
            fprintf(stderr, "Invalid data format on line %d: Missing label. Skipping.\n", line_number);
            continue; // Skip this data point
        }

        int label = atoi(label_str);
        if (label < 0 || label >= 6) {
            fprintf(stderr, "Invalid label %d on line %d. Skipping.\n", label, line_number);
            continue; // Skip this data point
        }

        // Extract text
        size_t text_length = last_comma - line;
        if (text_length >= MAX_TEXT_LENGTH) {
            fprintf(stderr, "Text too long on line %d. Skipping.\n", line_number);
            continue; // Skip this data point
        }

        char text_buffer[MAX_TEXT_LENGTH];
        strncpy(text_buffer, line, text_length);
        text_buffer[text_length] = '\0'; // Null-terminate

        // Remove surrounding quotes if present
        if (text_buffer[0] == '"' && text_buffer[text_length - 1] == '"') {
            text_buffer[text_length - 1] = '\0';
            memmove(text_buffer, text_buffer + 1, text_length - 1);
        }

        // Store the data point
        strncpy(data[*num_datapoints].text, text_buffer, MAX_TEXT_LENGTH - 1);
        data[*num_datapoints].text[MAX_TEXT_LENGTH - 1] = '\0'; // Ensure null-termination
        data[*num_datapoints].label = label;

        (*num_datapoints)++;

        // If we've reached capacity, realloc to increase size
        if (*num_datapoints >= capacity) {
            capacity *= 2;
            DataPoint* temp = realloc(data, capacity * sizeof(DataPoint));
            if (!temp) {
                perror("Reallocation failed in parseCSV");
                free(data);
                fclose(fp);
                return NULL;
            }
            data = temp;
        }
    }

    fclose(fp);

    // Optionally, shrink the allocated memory to fit the actual number of data points
    if (*num_datapoints < capacity) {
        DataPoint* temp = realloc(data, (*num_datapoints) * sizeof(DataPoint));
        if (temp) {
            data = temp;
        }
        // If realloc fails, continue with the larger memory block
    }

    return data;
}

void freeData(DataPoint* data, int num_datapoints) {
    if (data) {
        free(data);
    }
}