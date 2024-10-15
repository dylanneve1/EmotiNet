#ifndef NETWORK_H
#define NETWORK_H

#include <stdio.h>
#include <stdlib.h>

// Structure for a Neural Network
typedef struct {
    int input_nodes;
    int hidden_nodes;
    int output_nodes;
    float **weights_ih; // Weights from Input to Hidden layer
    float **weights_ho; // Weights from Hidden to Output layer
    float *hidden_bias;
    float *output_bias;
} NeuralNetwork;

// Function prototypes
NeuralNetwork* createNetwork(int input_nodes, int hidden_nodes, int output_nodes);
void train(NeuralNetwork* nn, float **inputs, float **targets, int num_samples, float learning_rate, int epochs);
float* predict(NeuralNetwork *nn, float *inputs);
void freeNetwork(NeuralNetwork* nn);

// Activation functions
float sigmoid(float x);
float sigmoid_derivative(float x);

// Model serialization functions (Binary Format)
int saveNetworkBinary(NeuralNetwork *nn, char **vocab, int vocab_size, const char* filename);
NeuralNetwork* loadNetworkBinary(const char* filename, char ***vocab, int *vocab_size);

#endif