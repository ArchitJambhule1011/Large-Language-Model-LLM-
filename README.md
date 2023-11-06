# Transformer-based Language Model

This repository contains an implementation of a transformer-based language model using PyTorch. This model is capable of generating text and estimating loss for a given text dataset. It is demonstrated using a simple character-level language modeling task.
Table of Contents

    1. Overview
    2. Getting Started
    3. Data Preparation
    4. Model Architecture
    5. Training
    6. Generating Text
    7. License

## Overview

The code provided in this repository implements a character-level language model using a transformer-based architecture. The transformer model is a state-of-the-art architecture for various natural language processing tasks, including language modeling, machine translation, and more.
Getting Started

To get started, you will need to install PyTorch and other necessary dependencies. 

You can also use a GPU if it's available to accelerate training and generation. The code will automatically switch between CPU and GPU based on the availability of a CUDA-compatible device.
Data Preparation

The code assumes you have a text dataset in a file named "Tiny_shakespeare.txt." You can replace this file with your own dataset if desired. The code will read the text and prepare it for training and evaluation.

## Model Architecture

The implemented model is a simple character-level language model that uses a transformer architecture. The model consists of an embedding layer followed by a transformer encoder. It is defined in the Bigram_model class.

## Training

The code trains the language model by estimating loss and updating model parameters using the AdamW optimizer. Training can be configured by adjusting the following hyperparameters:

    batch_size: Batch size for training.
    block_size: The length of input sequences.
    max_iters: The maximum number of training iterations.
    eval_interval: Interval for evaluating and printing training and validation loss.
    learning_rate: Learning rate for the optimizer.

To start training, simply run the code, and it will print training and validation loss at the specified intervals.


## Generating Text

After training, you can use the model to generate text. The code provides a function generate to generate text given an initial input. You can specify the number of new tokens to generate by setting max_new_tokens.

python

id = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decode(m.generate(id, max_new_tokens=500)[0].tolist())

```bash
   id = torch.zeros((1, 1), dtype=torch.long, device=device)
   generated_text = decode(m.generate(id, max_new_tokens=500)[0].tolist())
   ```

This will generate text based on the trained model, starting from an initial input.

## License

This code is available under the MIT License. You are free to use and modify it for your own projects. Please refer to the LICENSE file for more details.