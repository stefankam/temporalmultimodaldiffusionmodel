# Temporal Multimodal Diffusion Model

## Overview

This repository contains the code for implementing a temporal diffusion model for multimodal data, which aims to propagate information through time steps while considering both image and text data. In this README, we will describe the mathematical model and explain how the dimensions should match.

## Mathematical Model

Let's describe the mathematical model step by step:

### Input Data

**Image Data:** \(X_i\), where \(i\) represents the image index in the sequence. \(X_i\) has shape \((256,256,1)\) for each image.

**Text Data:** \(T_i\), where \(i\) represents the text index in the sequence. \(T_i\) has shape \((10,)\) for each text sequence.

### Temporal Modeling Using LSTM

The LSTM layer is applied to the concatenated multimodal features, aiming to capture temporal dependencies across time steps. Let \(H_t\) represent the hidden state of the LSTM at time step \(t\). The LSTM equations are as follows:

\[H_t = LSTM(H_{t-1}, [X_t, T_t])\]

where \([X_t, T_t]\) represents the concatenated features at time step \(t\). \(H_t\) has shape \((128,)\) since we have defined LSTM(128).

Here's how the LSTM layer works in this context:

- Temporal Modeling Within Each Modality (Images and Text):

For images, the LSTM layer captures temporal dependencies within the image sequence. If we have a sequence of images over time (e.g., frames from a video), the LSTM layer can capture how information in the images evolves from one frame to the next.

For text, the LSTM layer captures temporal dependencies within the text sequence. It can capture how the meaning of a sentence or sequence of words changes over time, especially in cases where text data has a time-based order.

- Concatenation of Features at Each Time Step:

Before passing the data through the LSTM layer, the code concatenates the features from both modalities (image and text) at each time step. This means that at each time step, the LSTM receives a combined feature vector that includes information from both the image and text.

The concatenation is performed to allow the model to capture interactions and dependencies between image and text data at each time step.
However, it's important to note that the provided code does not explicitly model temporal dependencies that might exist between different time steps in the text or how text and image data interact over time. To capture such interactions, we would need a more complex multimodal architecture that considers cross-modal dependencies and temporal relationships explicitly.

### Output Layer

The output layer produces a single sigmoid output representing the prediction. Let \(Y_t\) be the predicted output at time step \(t\).

Yt=Dense(Ht)Y_t = Dense(H_t)

where `Dense` represents the dense layer with a sigmoid activation function. \(Y_t\) has shape \((1,)\).

## Matching Dimensions

In the code, we reshape the merged_output to have shape \((1,2)\) before feeding it into the LSTM layer. This matches the shape requirement for the LSTM layer. The LSTM layer then outputs \((128,)\) at each time step, capturing temporal information. The output layer takes the temporal output and produces \((1,)\), which is the final prediction for each time step.

## Usage

To use this code for your own temporal multimodal diffusion model, follow the instructions below:

1. Clone this repository to your local machine.
2. Install the required dependencies.
3. Replace the sample data and descriptions with your own image and text data.
4. Modify the code as needed to match the dimensions and characteristics of your data.
5. Train and evaluate the model using your data.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

