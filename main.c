// main.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "./dataParsing/dataParser.h"
#include "./network/network.h"
#include "./dataParsing/vocabHash.h"

// Define emotion labels corresponding to their numerical indices
const char* emotion_labels[6] = {
    "Sadness",
    "Joy",
    "Love",
    "Anger",
    "Fear",
    "Surprise"
};

// Function to convert text to numerical input (Bag of Words)
float* textToInput(const char* text, int vocab_size, char **vocab) {
    float* input = (float*)calloc(vocab_size, sizeof(float)); // Initialize with zeros
    if (!input) {
        perror("Memory allocation failed in textToInput");
        return NULL;
    }

    // Make a copy of the text to tokenize
    char *text_copy = strdup(text);
    if (!text_copy) {
        perror("Memory allocation failed for text_copy");
        free(input);
        return NULL;
    }

    // Tokenize the text based on delimiters
    char *token = strtok(text_copy, " \t\n\r.,;!?\"'");
    while (token != NULL) {
        // Convert token to lowercase for case-insensitive matching
        for (int i = 0; token[i]; i++) {
            token[i] = tolower(token[i]);
        }

        // Linear search in the vocabulary
        for (int i = 0; i < vocab_size; i++) {
            if (strcmp(token, vocab[i]) == 0) {
                input[i] += 1.0f; // Increment the count for this word
                break;
            }
        }

        token = strtok(NULL, " \t\n\r.,;!?\"'");
    }

    free(text_copy);
    return input;
}

// Function to build a vocabulary using a hash table (uthash)
char** buildVocabulary(DataPoint* data, int num_datapoints, int *vocab_size) {
    VocabEntry *hash_table = NULL; // Initialize the hash table

    for (int i = 0; i < num_datapoints; i++) {
        // Make a copy of the text to tokenize
        char *text_copy = strdup(data[i].text);
        if (!text_copy) {
            perror("Memory allocation failed for text_copy in buildVocabulary");
            // Free hash table
            VocabEntry *current_entry, *tmp;
            HASH_ITER(hh, hash_table, current_entry, tmp) {
                HASH_DEL(hash_table, current_entry);
                free(current_entry->word);
                free(current_entry);
            }
            return NULL;
        }

        // Tokenize the text based on delimiters
        char *token = strtok(text_copy, " \t\n\r.,;!?\"'");
        while (token != NULL) {
            // Convert token to lowercase for case-insensitive matching
            for (int j = 0; token[j]; j++) {
                token[j] = tolower(token[j]);
            }

            // Check if the word is already in the hash table
            VocabEntry *entry;
            HASH_FIND_STR(hash_table, token, entry);
            if (entry == NULL) {
                // Add new word to the hash table
                entry = (VocabEntry*)malloc(sizeof(VocabEntry));
                if (!entry) {
                    perror("Memory allocation failed for VocabEntry");
                    free(text_copy);
                    // Free hash table
                    VocabEntry *current_entry, *tmp;
                    HASH_ITER(hh, hash_table, current_entry, tmp) {
                        HASH_DEL(hash_table, current_entry);
                        free(current_entry->word);
                        free(current_entry);
                    }
                    return NULL;
                }
                entry->word = strdup(token);
                if (!entry->word) {
                    perror("Memory allocation failed for vocab word");
                    free(entry);
                    free(text_copy);
                    // Free hash table
                    VocabEntry *current_entry, *tmp;
                    HASH_ITER(hh, hash_table, current_entry, tmp) {
                        HASH_DEL(hash_table, current_entry);
                        free(current_entry->word);
                        free(current_entry);
                    }
                    return NULL;
                }
                HASH_ADD_STR(hash_table, word, entry);
            }
            token = strtok(NULL, " \t\n\r.,;!?\"'");
        }

        free(text_copy);
    }

    // Now, extract the vocabulary from the hash table into an array
    *vocab_size = HASH_COUNT(hash_table);
    char **vocab = (char**)malloc((*vocab_size) * sizeof(char*));
    if (!vocab) {
        perror("Memory allocation failed for vocab array");
        // Free hash table
        VocabEntry *current_entry, *tmp;
        HASH_ITER(hh, hash_table, current_entry, tmp) {
            HASH_DEL(hash_table, current_entry);
            free(current_entry->word);
            free(current_entry);
        }
        return NULL;
    }

    int index = 0;
    VocabEntry *current_entry, *tmp;
    HASH_ITER(hh, hash_table, current_entry, tmp) {
        vocab[index++] = strdup(current_entry->word);
        // Free hash table entries
        HASH_DEL(hash_table, current_entry);
        free(current_entry->word);
        free(current_entry);
    }

    return vocab;
}

int main() {
    int choice;
    NeuralNetwork* nn = NULL;
    char **vocab = NULL;
    int vocab_size = 0;
    const char* model_filename = "model.bin"; // Binary model file

    printf("=== Emotion Classifier ===\n");
    printf("1. Load existing model\n");
    printf("2. Train a new model\n");
    printf("Choose an option (1 or 2): ");
    if (scanf("%d", &choice) != 1) {
        fprintf(stderr, "Invalid input. Exiting.\n");
        return 1;
    }
    getchar(); // Consume the newline character after the number

    if (choice == 1) {
        // Load existing model (binary format)
        nn = loadNetworkBinary(model_filename, &vocab, &vocab_size);
        if (!nn) {
            fprintf(stderr, "Failed to load the model. Exiting.\n");
            return 1;
        }
        printf("Model loaded successfully from '%s'.\n", model_filename);
    }
    else if (choice == 2) {
        // Train a new model
        int num_datapoints;
        DataPoint* data = parseCSV("emotions.csv", &num_datapoints);

        if (!data) {
            fprintf(stderr, "Error parsing CSV file.\n");
            return 1;
        }

        printf("Total valid data points: %d\n", num_datapoints);

        // Build vocabulary using hash table
        vocab = buildVocabulary(data, num_datapoints, &vocab_size);
        if (!vocab) {
            fprintf(stderr, "Failed to build vocabulary.\n");
            freeData(data, num_datapoints);
            return 1;
        }

        printf("Vocabulary size: %d\n", vocab_size);

        // Determine input size (size of vocabulary)
        int input_size = vocab_size;

        // Allocate memory for inputs and targets
        float** inputs = (float**)malloc(num_datapoints * sizeof(float*));
        float** targets = (float**)malloc(num_datapoints * sizeof(float*));

        if (!inputs || !targets) {
            fprintf(stderr, "Memory allocation failed for inputs or targets.\n");
            // Free allocated memory
            if (inputs) free(inputs);
            if (targets) free(targets);
            for (int i = 0; i < vocab_size; i++) {
                free(vocab[i]);
            }
            free(vocab);
            freeData(data, num_datapoints);
            return 1;
        }

        // Convert text data to numerical input and create target arrays
        for (int i = 0; i < num_datapoints; i++) {
            inputs[i] = textToInput(data[i].text, vocab_size, vocab);
            if (!inputs[i]) {
                fprintf(stderr, "Failed to convert text to input for data point %d.\n", i);
                // Free previously allocated inputs and targets
                for (int j = 0; j < i; j++) {
                    free(inputs[j]);
                    free(targets[j]);
                }
                free(inputs);
                free(targets);
                for (int j = 0; j < vocab_size; j++) {
                    free(vocab[j]);
                }
                free(vocab);
                freeData(data, num_datapoints);
                return 1;
            }

            targets[i] = (float*)calloc(6, sizeof(float)); // 6 output nodes for 6 emotions
            if (!targets[i]) {
                fprintf(stderr, "Memory allocation failed for target of data point %d.\n", i);
                // Free previously allocated inputs and targets
                for (int j = 0; j <= i; j++) {
                    free(inputs[j]);
                    if (j < i) free(targets[j]);
                }
                free(inputs);
                free(targets);
                for (int j = 0; j < vocab_size; j++) {
                    free(vocab[j]);
                }
                free(vocab);
                freeData(data, num_datapoints);
                return 1;
            }

            // One-hot encoding for targets
            targets[i][data[i].label] = 1.0f;
        }

        // Free the raw data as it's no longer needed
        freeData(data, num_datapoints);

        // 2. Create and Train the Network
        int hidden_nodes = 10;       // Example: Adjust as needed
        float learning_rate = 0.1f; // Example: Adjust as needed
        int epochs = 100;            // Example: Adjust as needed

        nn = createNetwork(input_size, hidden_nodes, 6); // 6 output nodes for 6 emotions
        if (!nn) {
            fprintf(stderr, "Failed to create neural network.\n");
            // Free inputs, targets, vocab
            for (int i = 0; i < num_datapoints; i++) {
                free(inputs[i]);
                free(targets[i]);
            }
            free(inputs);
            free(targets);
            for (int j = 0; j < vocab_size; j++) {
                free(vocab[j]);
            }
            free(vocab);
            return 1;
        }

        printf("Training the neural network...\n");
        // Train the network
        train(nn, inputs, targets, num_datapoints, learning_rate, epochs);
        printf("Training completed.\n");

        // Save the model in binary format
        if (saveNetworkBinary(nn, vocab, vocab_size, model_filename)) {
            printf("Model saved successfully to '%s'.\n", model_filename);
        }
        else {
            fprintf(stderr, "Failed to save the model.\n");
        }

        // Free training data
        for (int i = 0; i < num_datapoints; i++) {
            free(inputs[i]);
            free(targets[i]);
        }
        free(inputs);
        free(targets);
    }
    else {
        fprintf(stderr, "Invalid choice. Exiting.\n");
        return 1;
    }

    // Interactive Classification Loop
    while (1) {
        char input_text[1024]; // Increased buffer size to handle longer inputs
        printf("\nEnter text to classify (or type 'exit' to quit):\n> ");
        if (!fgets(input_text, sizeof(input_text), stdin)) {
            fprintf(stderr, "Error reading input. Exiting.\n");
            break;
        }

        // Remove newline character
        input_text[strcspn(input_text, "\n")] = '\0';

        // Check for exit command
        if (strcasecmp(input_text, "exit") == 0) {
            printf("Exiting the program.\n");
            break;
        }

        // Convert input text to numerical input
        float* numerical_input = textToInput(input_text, vocab_size, vocab);
        if (!numerical_input) {
            fprintf(stderr, "Failed to convert input text to numerical format.\n");
            continue;
        }

        // Predict
        float *prediction = predict(nn, numerical_input);
        if (!prediction) {
            fprintf(stderr, "Prediction failed.\n");
            free(numerical_input);
            continue;
        }

        // Display the results with emotion names
        printf("Prediction:\n");
        for (int i = 0; i < 6; i++) {
            printf("%s: %f\n", emotion_labels[i], prediction[i]);
        }

        // Find the emotion with the highest score
        int predicted_emotion = 0;
        float max_score = prediction[0];
        for (int i = 1; i < 6; i++) {
            if (prediction[i] > max_score) {
                max_score = prediction[i];
                predicted_emotion = i;
            }
        }
        printf("Predicted Emotion: %s\n", emotion_labels[predicted_emotion]);

        // Free allocated memory
        free(numerical_input);
        free(prediction);
    }

    // Free allocated memory for vocabulary
    for (int i = 0; i < vocab_size; i++) {
        free(vocab[i]);
    }
    free(vocab);

    // Free neural network resources
    freeNetwork(nn);

    return 0;
}