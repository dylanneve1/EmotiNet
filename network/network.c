#include "network.h"
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Sigmoid activation function
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Derivative of the sigmoid function
float sigmoid_derivative(float x) {
    return x * (1.0f - x);
}

// Helper function to multiply matrix and vector
void matrixVectorMultiply(float *result, float **matrix, float *vector, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        result[i] = 0.0f;
        for (int j = 0; j < cols; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
}

// Create a new neural network
NeuralNetwork* createNetwork(int input_nodes, int hidden_nodes, int output_nodes) {
    NeuralNetwork* nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (!nn) {
        perror("Memory allocation failed for NeuralNetwork");
        return NULL;
    }

    srand((unsigned int)time(NULL)); // Seed for random number generation

    nn->input_nodes = input_nodes;
    nn->hidden_nodes = hidden_nodes;
    nn->output_nodes = output_nodes;

    // Allocate weights from Input to Hidden
    nn->weights_ih = (float**)malloc(hidden_nodes * sizeof(float*));
    if (!nn->weights_ih) {
        perror("Memory allocation failed for weights_ih");
        free(nn);
        return NULL;
    }

    for (int i = 0; i < hidden_nodes; i++) {
        nn->weights_ih[i] = (float*)malloc(input_nodes * sizeof(float));
        if (!nn->weights_ih[i]) {
            perror("Memory allocation failed for weights_ih row");
            // Free previously allocated rows
            for (int j = 0; j < i; j++) {
                free(nn->weights_ih[j]);
            }
            free(nn->weights_ih);
            free(nn);
            return NULL;
        }
        for (int j = 0; j < input_nodes; j++) {
            nn->weights_ih[i][j] = 2.0f * ((float)rand() / RAND_MAX) - 1.0f; // Initialize weights between -1 and 1
        }
    }

    // Allocate weights from Hidden to Output
    nn->weights_ho = (float**)malloc(output_nodes * sizeof(float*));
    if (!nn->weights_ho) {
        perror("Memory allocation failed for weights_ho");
        // Free weights_ih
        for (int i = 0; i < hidden_nodes; i++) {
            free(nn->weights_ih[i]);
        }
        free(nn->weights_ih);
        free(nn);
        return NULL;
    }

    for (int i = 0; i < output_nodes; i++) {
        nn->weights_ho[i] = (float*)malloc(hidden_nodes * sizeof(float));
        if (!nn->weights_ho[i]) {
            perror("Memory allocation failed for weights_ho row");
            // Free previously allocated rows
            for (int j = 0; j < i; j++) {
                free(nn->weights_ho[j]);
            }
            free(nn->weights_ho);
            // Free weights_ih
            for (int j = 0; j < hidden_nodes; j++) {
                free(nn->weights_ih[j]);
            }
            free(nn->weights_ih);
            free(nn);
            return NULL;
        }
        for (int j = 0; j < hidden_nodes; j++) {
            nn->weights_ho[i][j] = 2.0f * ((float)rand() / RAND_MAX) - 1.0f; // Initialize weights between -1 and 1
        }
    }

    // Allocate and initialize biases
    nn->hidden_bias = (float*)malloc(hidden_nodes * sizeof(float));
    if (!nn->hidden_bias) {
        perror("Memory allocation failed for hidden_bias");
        // Free weights_ho and weights_ih
        for (int i = 0; i < output_nodes; i++) {
            free(nn->weights_ho[i]);
        }
        free(nn->weights_ho);
        for (int i = 0; i < hidden_nodes; i++) {
            free(nn->weights_ih[i]);
        }
        free(nn->weights_ih);
        free(nn);
        return NULL;
    }

    for (int i = 0; i < hidden_nodes; i++) {
        nn->hidden_bias[i] = 2.0f * ((float)rand() / RAND_MAX) - 1.0f; // Initialize biases between -1 and 1
    }

    nn->output_bias = (float*)malloc(output_nodes * sizeof(float));
    if (!nn->output_bias) {
        perror("Memory allocation failed for output_bias");
        // Free hidden_bias, weights_ho and weights_ih
        free(nn->hidden_bias);
        for (int i = 0; i < output_nodes; i++) {
            free(nn->weights_ho[i]);
        }
        free(nn->weights_ho);
        for (int i = 0; i < hidden_nodes; i++) {
            free(nn->weights_ih[i]);
        }
        free(nn->weights_ih);
        free(nn);
        return NULL;
    }

    for (int i = 0; i < output_nodes; i++) {
        nn->output_bias[i] = 2.0f * ((float)rand() / RAND_MAX) - 1.0f; // Initialize biases between -1 and 1
    }

    return nn;
}

// Train the neural network using Backpropagation
void train(NeuralNetwork* nn, float **inputs, float **targets, int num_samples, float learning_rate, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_error = 0.0f;

        for (int sample = 0; sample < num_samples; sample++) {
            // ----- Feedforward -----
            // Calculate Hidden Layer Activations
            float hidden_inputs[nn->hidden_nodes];
            matrixVectorMultiply(hidden_inputs, nn->weights_ih, inputs[sample], nn->hidden_nodes, nn->input_nodes);
            for (int i = 0; i < nn->hidden_nodes; i++) {
                hidden_inputs[i] += nn->hidden_bias[i];
                hidden_inputs[i] = sigmoid(hidden_inputs[i]);
            }

            // Calculate Output Layer Activations
            float output_inputs[nn->output_nodes];
            matrixVectorMultiply(output_inputs, nn->weights_ho, hidden_inputs, nn->output_nodes, nn->hidden_nodes);
            for (int i = 0; i < nn->output_nodes; i++) {
                output_inputs[i] += nn->output_bias[i];
                output_inputs[i] = sigmoid(output_inputs[i]);
            }

            // ----- Calculate Error -----
            float output_errors[nn->output_nodes];
            for (int i = 0; i < nn->output_nodes; i++) {
                output_errors[i] = targets[sample][i] - output_inputs[i];
                total_error += output_errors[i] * output_errors[i];
            }

            // ----- Backpropagation -----
            // Calculate gradients for output layer
            float output_gradients[nn->output_nodes];
            for (int i = 0; i < nn->output_nodes; i++) {
                output_gradients[i] = output_errors[i] * sigmoid_derivative(output_inputs[i]);
            }

            // Calculate errors for hidden layer
            float hidden_errors[nn->hidden_nodes];
            for (int i = 0; i < nn->hidden_nodes; i++) {
                hidden_errors[i] = 0.0f;
                for (int j = 0; j < nn->output_nodes; j++) {
                    hidden_errors[i] += nn->weights_ho[j][i] * output_errors[j];
                }
            }

            // Calculate gradients for hidden layer
            float hidden_gradients[nn->hidden_nodes];
            for (int i = 0; i < nn->hidden_nodes; i++) {
                hidden_gradients[i] = hidden_errors[i] * sigmoid_derivative(hidden_inputs[i]);
            }

            // ----- Update Weights and Biases -----
            // Update weights from Hidden to Output
            for (int i = 0; i < nn->output_nodes; i++) {
                for (int j = 0; j < nn->hidden_nodes; j++) {
                    nn->weights_ho[i][j] += learning_rate * output_gradients[i] * hidden_inputs[j];
                }
                nn->output_bias[i] += learning_rate * output_gradients[i];
            }

            // Update weights from Input to Hidden
            for (int i = 0; i < nn->hidden_nodes; i++) {
                for (int j = 0; j < nn->input_nodes; j++) {
                    nn->weights_ih[i][j] += learning_rate * hidden_gradients[i] * inputs[sample][j];
                }
                nn->hidden_bias[i] += learning_rate * hidden_gradients[i];
            }
        }

        // Calculate Mean Squared Error for the epoch
        float mse = total_error / num_samples;
        printf("Epoch %d/%d, MSE: %f\n", epoch + 1, epochs, mse);
    }
}

// Predict output (Feedforward)
float* predict(NeuralNetwork *nn, float *inputs) {
    float *hidden_outputs = (float*)malloc(nn->hidden_nodes * sizeof(float));
    if (!hidden_outputs) {
        perror("Memory allocation failed for hidden_outputs in predict");
        return NULL;
    }

    matrixVectorMultiply(hidden_outputs, nn->weights_ih, inputs, nn->hidden_nodes, nn->input_nodes);
    for (int i = 0; i < nn->hidden_nodes; i++) {
        hidden_outputs[i] += nn->hidden_bias[i];
        hidden_outputs[i] = sigmoid(hidden_outputs[i]);
    }

    float *outputs = (float*)malloc(nn->output_nodes * sizeof(float));
    if (!outputs) {
        perror("Memory allocation failed for outputs in predict");
        free(hidden_outputs);
        return NULL;
    }

    matrixVectorMultiply(outputs, nn->weights_ho, hidden_outputs, nn->output_nodes, nn->hidden_nodes);
    for (int i = 0; i < nn->output_nodes; i++) {
        outputs[i] += nn->output_bias[i];
        outputs[i] = sigmoid(outputs[i]);
    }

    free(hidden_outputs);
    return outputs; // Caller must free this memory!
}

// Free the neural network memory
void freeNetwork(NeuralNetwork* nn) {
    if (!nn) return;

    for (int i = 0; i < nn->hidden_nodes; i++) {
        free(nn->weights_ih[i]);
    }
    free(nn->weights_ih);

    for (int i = 0; i < nn->output_nodes; i++) {
        free(nn->weights_ho[i]);
    }
    free(nn->weights_ho);

    free(nn->hidden_bias);
    free(nn->output_bias);
    free(nn);
}

// Save the neural network and vocabulary to a proprietary binary file
int saveNetworkBinary(NeuralNetwork *nn, char **vocab, int vocab_size, const char* filename) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror("Failed to open file for saving network in binary format");
        return 0;
    }

    // Define a magic number and version for file validation
    const char magic_number[8] = "EMOTIONN"; // 8 bytes
    uint32_t version = 1;

    // Write magic number
    if (fwrite(magic_number, sizeof(char), 8, fp) != 8) {
        fprintf(stderr, "Failed to write magic number.\n");
        fclose(fp);
        return 0;
    }

    // Write version number
    if (fwrite(&version, sizeof(uint32_t), 1, fp) != 1) {
        fprintf(stderr, "Failed to write version number.\n");
        fclose(fp);
        return 0;
    }

    // Write network architecture
    if (fwrite(&(nn->input_nodes), sizeof(int), 1, fp) != 1 ||
        fwrite(&(nn->hidden_nodes), sizeof(int), 1, fp) != 1 ||
        fwrite(&(nn->output_nodes), sizeof(int), 1, fp) != 1 ||
        fwrite(&vocab_size, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "Failed to write network architecture.\n");
        fclose(fp);
        return 0;
    }

    // Write vocabulary
    for (int i = 0; i < vocab_size; i++) {
        uint32_t word_length = strlen(vocab[i]);
        // Write the length of the word
        if (fwrite(&word_length, sizeof(uint32_t), 1, fp) != 1) {
            fprintf(stderr, "Failed to write word length for vocab index %d.\n", i);
            fclose(fp);
            return 0;
        }
        // Write the word characters
        if (fwrite(vocab[i], sizeof(char), word_length, fp) != word_length) {
            fprintf(stderr, "Failed to write word for vocab index %d.\n", i);
            fclose(fp);
            return 0;
        }
    }

    // Write weights_ih
    for (int i = 0; i < nn->hidden_nodes; i++) {
        if (fwrite(nn->weights_ih[i], sizeof(float), nn->input_nodes, fp) != (size_t)nn->input_nodes) {
            fprintf(stderr, "Failed to write weights_ih for hidden node %d.\n", i);
            fclose(fp);
            return 0;
        }
    }

    // Write weights_ho
    for (int i = 0; i < nn->output_nodes; i++) {
        if (fwrite(nn->weights_ho[i], sizeof(float), nn->hidden_nodes, fp) != (size_t)nn->hidden_nodes) {
            fprintf(stderr, "Failed to write weights_ho for output node %d.\n", i);
            fclose(fp);
            return 0;
        }
    }

    // Write hidden_bias
    if (fwrite(nn->hidden_bias, sizeof(float), nn->hidden_nodes, fp) != (size_t)nn->hidden_nodes) {
        fprintf(stderr, "Failed to write hidden_bias.\n");
        fclose(fp);
        return 0;
    }

    // Write output_bias
    if (fwrite(nn->output_bias, sizeof(float), nn->output_nodes, fp) != (size_t)nn->output_nodes) {
        fprintf(stderr, "Failed to write output_bias.\n");
        fclose(fp);
        return 0;
    }

    fclose(fp);
    return 1;
}

// Load the neural network and vocabulary from a proprietary binary file
NeuralNetwork* loadNetworkBinary(const char* filename, char ***vocab, int *vocab_size) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror("Failed to open file for loading network in binary format");
        return NULL;
    }

    // Read and validate magic number
    char magic_number[8];
    if (fread(magic_number, sizeof(char), 8, fp) != 8) {
        fprintf(stderr, "Failed to read magic number.\n");
        fclose(fp);
        return NULL;
    }

    if (strncmp(magic_number, "EMOTIONN", 8) != 0) {
        fprintf(stderr, "Invalid magic number. Not a valid EMOTIONN model file.\n");
        fclose(fp);
        return NULL;
    }

    // Read and validate version number
    uint32_t version;
    if (fread(&version, sizeof(uint32_t), 1, fp) != 1) {
        fprintf(stderr, "Failed to read version number.\n");
        fclose(fp);
        return NULL;
    }

    if (version != 1) {
        fprintf(stderr, "Unsupported version number: %u.\n", version);
        fclose(fp);
        return NULL;
    }

    // Read network architecture
    int input_nodes, hidden_nodes, output_nodes;
    if (fread(&input_nodes, sizeof(int), 1, fp) != 1 ||
        fread(&hidden_nodes, sizeof(int), 1, fp) != 1 ||
        fread(&output_nodes, sizeof(int), 1, fp) != 1 ||
        fread(vocab_size, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "Failed to read network architecture.\n");
        fclose(fp);
        return NULL;
    }

    // Allocate and read vocabulary
    *vocab = (char**)malloc((*vocab_size) * sizeof(char*));
    if (!(*vocab)) {
        perror("Memory allocation failed for vocabulary");
        fclose(fp);
        return NULL;
    }

    for (int i = 0; i < *vocab_size; i++) {
        uint32_t word_length;
        if (fread(&word_length, sizeof(uint32_t), 1, fp) != 1) {
            fprintf(stderr, "Failed to read word length for vocab index %d.\n", i);
            // Free previously allocated vocab
            for (int j = 0; j < i; j++) {
                free((*vocab)[j]);
            }
            free(*vocab);
            fclose(fp);
            return NULL;
        }

        // Allocate memory for the word plus a null terminator
        (*vocab)[i] = (char*)malloc((word_length + 1) * sizeof(char));
        if (!(*vocab)[i]) {
            perror("Memory allocation failed for vocab word");
            // Free previously allocated vocab
            for (int j = 0; j < i; j++) {
                free((*vocab)[j]);
            }
            free(*vocab);
            fclose(fp);
            return NULL;
        }

        // Read the word characters
        if (fread((*vocab)[i], sizeof(char), word_length, fp) != word_length) {
            fprintf(stderr, "Failed to read word for vocab index %d.\n", i);
            // Free previously allocated vocab
            for (int j = 0; j <= i; j++) {
                free((*vocab)[j]);
            }
            free(*vocab);
            fclose(fp);
            return NULL;
        }

        // Null-terminate the string
        (*vocab)[i][word_length] = '\0';
    }

    // Create the network
    NeuralNetwork *nn = createNetwork(input_nodes, hidden_nodes, output_nodes);
    if (!nn) {
        // Free vocabulary
        for (int i = 0; i < *vocab_size; i++) {
            free((*vocab)[i]);
        }
        free(*vocab);
        fclose(fp);
        return NULL;
    }

    // Read weights_ih
    for (int i = 0; i < hidden_nodes; i++) {
        if (fread(nn->weights_ih[i], sizeof(float), input_nodes, fp) != (size_t)input_nodes) {
            fprintf(stderr, "Failed to read weights_ih for hidden node %d.\n", i);
            freeNetwork(nn);
            for (int k = 0; k < *vocab_size; k++) {
                free((*vocab)[k]);
            }
            free(*vocab);
            fclose(fp);
            return NULL;
        }
    }

    // Read weights_ho
    for (int i = 0; i < output_nodes; i++) {
        if (fread(nn->weights_ho[i], sizeof(float), hidden_nodes, fp) != (size_t)hidden_nodes) {
            fprintf(stderr, "Failed to read weights_ho for output node %d.\n", i);
            freeNetwork(nn);
            for (int k = 0; k < *vocab_size; k++) {
                free((*vocab)[k]);
            }
            free(*vocab);
            fclose(fp);
            return NULL;
        }
    }

    // Read hidden_bias
    if (fread(nn->hidden_bias, sizeof(float), hidden_nodes, fp) != (size_t)hidden_nodes) {
        fprintf(stderr, "Failed to read hidden_bias.\n");
        freeNetwork(nn);
        for (int k = 0; k < *vocab_size; k++) {
            free((*vocab)[k]);
        }
        free(*vocab);
        fclose(fp);
        return NULL;
    }

    // Read output_bias
    if (fread(nn->output_bias, sizeof(float), output_nodes, fp) != (size_t)output_nodes) {
        fprintf(stderr, "Failed to read output_bias.\n");
        freeNetwork(nn);
        for (int k = 0; k < *vocab_size; k++) {
            free((*vocab)[k]);
        }
        free(*vocab);
        fclose(fp);
        return NULL;
    }

    fclose(fp);
    return nn;
}