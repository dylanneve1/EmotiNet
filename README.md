# Emotion Classifier in C [EmotiNet]

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![C](https://img.shields.io/badge/language-C-green.svg)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
  - [Training a New Model](#training-a-new-model)
  - [Loading an Existing Model](#loading-an-existing-model)
  - [Interactive Classification](#interactive-classification)
- [Data Format](#data-format)
- [Project Structure](#project-structure)
- [Memory Considerations](#memory-considerations)
- [Contributing](#contributing)
- [License](#license)

## Overview

The **Emotion Classifier** is a C-based application that leverages a neural network to classify text inputs into one of six predefined emotions: Sadness, Joy, Love, Anger, Fear, and Surprise. It processes a dataset from a CSV file, builds a vocabulary, trains a neural network model, and provides an interactive interface for emotion prediction.

## Features

- **Neural Network Implementation:** Custom neural network built from scratch in C.
- **Modern Training Pipeline:** Uses ReLU activations with He initialization and a softmax output layer trained via cross-entropy loss for improved accuracy.
- **Vocabulary Building:** Efficiently constructs a vocabulary from the dataset using hash tables.
- **Model Persistence:** Saves and loads trained models in binary format.
- **Interactive Interface:** Allows users to input text and receive emotion predictions in real-time.
- **Memory Optimization:** Limits vocabulary size to manage memory usage effectively.
- **Error Handling:** Robust error checking and memory management to prevent crashes.

## Architecture

The project is structured into several modules, each responsible for specific functionalities:

- **main.c:** Entry point of the application. Handles user interactions, model training, and prediction.
- **network (subfolder):** Contains `network.c` and `network.h`, which implement the neural network with ReLU hidden layers and a softmax output trained via cross-entropy.
- **dataParsing (subfolder):** Contains `dataParser.c`, `dataParser.h`, `vocabHash.h`, which handle dataset parsing and vocabulary management.
- **Makefile:** Automates the build process, compiling source files and managing dependencies.

## Dependencies

- **C Compiler:** GCC or any compatible C compiler.
- **uthash:** A header-only C library for hash tables. Included in the `include/` directory.
- **Make:** For using the provided Makefile to build the project.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/dylanneve1/EmotiNet.git
   cd EmotiNet
   ```

2. **Ensure Dependencies Are Met:**

   - **GCC:** Verify installation.

     ```bash
     gcc --version
     ```

   - **Make:** Verify installation.

     ```bash
     make --version
     ```

   - **uthash:** Already included in the `include/` directory.

3. **Prepare the Dataset:**

   - Place your `emotions.csv` file in the project root directory.
   - Ensure the CSV is properly formatted with each line containing a text and a label separated by a comma. Example:

     ```
     "I am so happy and joyful today!",1
     "Feeling sad and low.",0
     ```

## Usage

### Training a New Model

1. **Run the Application:**

   ```bash
   ./main
   ```

2. **Select Training Option:**

   ```
   === Emotion Classifier ===
   1. Load existing model
   2. Train a new model
   Choose an option (1 or 2): 2
   ```

3. **Training Process:**

   The application will parse the CSV, build the vocabulary (limited to the top 10,000 words to manage memory), convert text data to numerical input, train the neural network, and save the model to `model.bin`.

   ```
   Total valid data points: 416808
   Vocabulary size: 10000
   Training the neural network...
   Training completed.
   Model saved successfully to 'model.bin'.
   ```

### Loading an Existing Model

1. **Run the Application:**

   ```bash
   ./main
   ```

2. **Select Loading Option:**

   ```
   === Emotion Classifier ===
   1. Load existing model
   2. Train a new model
   Choose an option (1 or 2): 1
   ```

3. **Model Loading:**

   The application will load the neural network model from `model.bin`.

   ```
   Model loaded successfully from 'model.bin'.
   ```

### Interactive Classification

After training or loading a model, the application enters an interactive loop where you can input text and receive emotion predictions.

```
Enter text to classify (or type 'exit' to quit):
> I am feeling very happy and excited today!
Prediction:
Sadness: 0.123456
Joy: 0.812345
Love: 0.054321
Anger: 0.067890
Fear: 0.045678
Surprise: 0.032109
Predicted Emotion: Joy

Enter text to classify (or type 'exit' to quit):
> exit
Exiting the program.
```

## Data Format

The `emotions.csv` should adhere to the following structure:

- **Fields:** Each line contains two fields: the text and the corresponding emotion label.
- **Delimiter:** Comma-separated.
- **Quotation:** Text containing commas should be enclosed in double quotes to prevent misparsing.

**Example:**

```
"I am so happy and joyful today!",1
"Feeling sad and low.",0
"Getting angry about the delays.",3
"Excited for the upcoming event!",5
"I am fearful of the unknown.",4
"Disgusted by the poor service.",2
```

**Emotion Labels:**

| Label | Emotion    |
|-------|------------|
| 0     | Sadness    |
| 1     | Joy        |
| 2     | Love       |
| 3     | Anger      |
| 4     | Fear       |
| 5     | Surprise   |

## Project Structure

```
EmotiNet/
├── include/
│   └── uthash.h          # uthash header file
├── main.c
├── network/
│   ├── network.c
│   └── network.h
├── dataParsing/
│   ├── dataParser.c
│   ├── dataParser.h
│   └── vocabHash.h
├── Makefile
├── model.bin             # Generated after training
├── emotions.csv          # Your dataset
└── README.md
```

- **include/uthash.h:** Hash table library used for efficient vocabulary management.
- **main.c:** Handles user interactions, model training, loading, and prediction.
- **network (subfolder):** Contains the neural network implementation files.
- **dataParsing (subfolder):** Handles CSV parsing and vocabulary creation.
- **Makefile:** Automates the build process.
- **model.bin:** Binary file storing the trained neural network model.
- **emotions.csv:** CSV dataset containing text samples and their corresponding emotion labels.
- **README.md:** Project documentation.

## Memory Considerations

Handling large datasets and extensive vocabularies can lead to high memory consumption. To mitigate this:

- **Limit Vocabulary Size:** The application restricts the vocabulary to the top 10,000 most frequent words. Adjust `MAX_VOCAB_SIZE` in `main.c` as needed based on your system's capabilities.
- **Efficient Data Structures:** Utilizes hash tables for O(1) word lookups, reducing processing time.
- **Memory Monitoring:** Use tools like `htop` or `valgrind` to monitor and profile memory usage during execution.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**

2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add Your Feature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

## License

This project is licensed under the [MIT License](LICENSE).
