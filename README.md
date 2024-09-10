---

# Experimenting with a Transformer

---

# Transformer-based Text Generation and Sequence Prediction

This repository implements a transformer-based model for text generation and sequence prediction using PyTorch. The project builds a transformer architecture from scratch, tokenizes input text, and trains the model using custom datasets for supervised text prediction tasks. The repository includes utilities for training, validation, and text generation with beam search and top-k sampling. Sadly my local machine isn't powerful enough to go over 50 batches, so the model doesn't perform as well as it should.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Model Architecture](#model-architecture)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Training and Evaluation](#training-and-evaluation)
7. [Text Generation](#text-generation)
8. [Hyperparameters](#hyperparameters)
9. [PC Specs](#pc-specs)
10. [Acknowledgements](#acknowledgements)

## Project Overview

This project implements a transformer model for sequence-to-sequence tasks, inspired by models like GPT and BERT. The model learns from a dataset of paired text inputs (source and target sequences) and can predict or generate sequences based on learned patterns.

### Key Features:
- Custom tokenizer and vocabulary builder with support for handling out-of-vocabulary tokens (`<UNK>`) and padding (`<PAD>`).
- Multi-head attention-based transformer architecture.
- Dynamic batching with padding for handling input sequences of varying lengths.
- Beam search and top-k sampling for text generation with configurable parameters like temperature and top-k filtering.
- Early stopping and learning rate scheduling for efficient training.

## Project Structure

```
.
├── dataset/
│   └── dataset.csv             # Input text data (source-target pairs)
├── modules/
│   ├── transformer_class.py    # Transformer model definition
│   ├── textdataset_class.py    # Dataset class for text data
│   ├── tools.py                # Utility functions (set_seed, tokenizer, vocab builder)
├── train.py                    # Main script for training, validation, and text generation
├── README.md                   # Project documentation
└── requirements.txt            # Dependencies
```

### Key Files:

- **`transformer_class.py`**: Implements the transformer model with encoder and decoder layers, including multi-head attention and positional encoding.
- **`textdataset_class.py`**: Defines the `TextDataset` class for loading and tokenizing text data, along with custom collate functions for dynamic batching.
- **`tools.py`**: Contains utility functions such as `set_seed`, `simple_tokenizer`, and `build_vocab`, as well as text generation functions like `generate_text_with_top_k_sampling`.
- **`train.py`**: The entry point for training the model, validating it on a dataset, and generating text. This file handles dataset loading, model initialization, training, and text generation using beam search and top-k sampling.

## Model Architecture

The transformer model is implemented following the original architecture proposed in the [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper by Vaswani et al.

### Components:
- **Multi-head attention**: Allows the model to jointly attend to information from different positions in the sequence, improving the ability to capture dependencies across distant tokens.
- **Positional encoding**: Injects information about the position of tokens in a sequence, since transformers do not inherently understand the order of tokens.
- **Feed-forward network**: A two-layer fully connected feed-forward network applied to each position of the sequence independently.
- **Layer normalization and dropout**: Helps regularize the model and prevent overfitting.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/claudiunderthehood/ExperimentingTransformer
   cd ExperimentingTransformer
   ```

2. **Install the required dependencies**:
   First, make sure you have [Python 3.8+](https://www.python.org/downloads/) installed. Then install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   The main dependencies include:
   - `torch`: PyTorch framework for building and training the transformer model.
   - `pandas`: For loading and preprocessing the dataset.
   - `sklearn`: For train-test split.
   - `numpy`: For numerical operations.

3. **Download the dataset**:
   Place your dataset (formatted as `input` and `chosen` columns in a CSV file) in the `dataset/` folder.

## Usage

### Training the Model

1. **Prepare the dataset**: The dataset should be a CSV file where:
   - The `input` column contains the input sequences (source).
   - The `chosen` column contains the target sequences (target).
   
   Example:
   ```
   input,chosen
   "This is an example input","This is an example output"
   "Another input","Another output"
   ```

2. **Train the model**:
   Run the `train.py` script to start training the model. By default, it will load the dataset, initialize the transformer model, and begin training.

   ```bash
   python train.py
   ```

   During training, the script will print the training and validation loss for each epoch. Early stopping is implemented to prevent overfitting.

3. **Monitor the training**:
   Training will continue until:
   - The model reaches the maximum number of epochs.
   - The validation loss stops improving for a certain number of consecutive epochs (patience).

### Training and Evaluation

To train and validate the model:
```bash
python train.py
```
This will train the transformer model and print out the training and validation loss at each epoch.

### Text Generation

Once the model is trained, you can use it to generate text. The `generate_text_with_top_k_sampling` function supports top-k sampling with optional temperature control.

Example of generating text from a seed sequence:
```bash
python train.py
```

Within the `train.py` file, you can specify the seed sequence and max generation length:
```python
start_sequence = ["Example", "Sequence"]
max_len = 20
generated_text = generate_text_with_top_k_sampling(model, start_sequence, max_len, word2idx, idx2word, device, temperature=1.0)
print("Generated text:", " ".join(generated_text))
```

### Example Output:
```text
Generated text: My name is ...
```

## Hyperparameters

The following hyperparameters control the architecture of the transformer model. These parameters can be adjusted depending on the dataset size, task complexity, and available computing power:

```python
SRC_VOCAB_SIZE: int = len(word2idx)  # Size of the source vocabulary, based on the input data.
TGT_VOCAB_SIZE: int = len(word2idx)  # Size of the target vocabulary, often identical to SRC_VOCAB_SIZE.
D_MODEL: int = 500  # Dimensionality of the embeddings and transformer layers. Higher values may improve accuracy at the cost of computational resources.
NUM_HEADS: int = 10  # Number of attention heads in multi-head attention. Increasing this can help the model learn different aspects of the data.
NUM_LAYERS: int = 10  # Number of transformer layers (both encoder and decoder). More layers improve learning capacity but require more computational power.
D_FF: int = 1032  # Dimensionality of the feed-forward network within the transformer. Higher values increase model capacity but also computational cost.
MAX_SEQ_LENGTH: int = 50  # Maximum length of input sequences. Sequences longer than this will be truncated.
DROPOUT: float = 0.3  # Dropout rate to prevent overfitting. A higher dropout rate can help in smaller datasets, but too much dropout may slow learning.
```

### Recommendations:
- **For large datasets**: You can increase `D_MODEL`, `NUM_LAYERS`, and `NUM_HEADS` to improve learning capacity.
- **For small datasets**: Consider reducing `D_MODEL` and `D_FF` to prevent overfitting and reduce computational requirements.
- **Training time**: More layers and heads will significantly increase the required training time, so adjust these values based on your hardware capabilities.

### PC Specs

As stated in the introduction, my machine wasn't capable of producing decent results because of lack of power from my components. I'll list the components I have on my PC so that it can be used as a reference:

- **CPU**: AMD Ryzen 7 5800X
- **RAM**: DDR4 32 GB
- **GPU**: RTX 4060Ti 8GB

## Acknowledgements

This project is inspired by the transformer architecture introduced in the [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper. The codebase uses PyTorch for building the model and training, leveraging modern deep learning techniques for sequence prediction and text generation.

Feel free to modify, extend, and experiment with this codebase!