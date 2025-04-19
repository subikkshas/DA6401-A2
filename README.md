# DA6401_DL_ASSIGNMENT_02_Subikksha

This repository contains the code for Assignment 2 of the course **Introduction to Deep Learning (DA6401)** offered by the **Department of Data Science and Artificial Intelligence, IIT Madras**.

## Problem Statement

This assignment focuses on building and experimenting with CNN-based image classifiers using a subset of the iNaturalist dataset.

### Part A: Training a CNN from Scratch

- Build a small CNN model consisting of 5 convolution layers.
- Each convolutional layer is followed by an activation function (e.g., ReLU) and a max-pooling layer.
- After 5 such blocks, add a dense layer and an output layer with 10 neurons.
- Make the architecture flexible by allowing changes in:
  - Number of filters
  - Filter size
  - Activation function
  - Dense layer size
- Calculate:
  - Total number of computations (based on filter size and neurons)
  - Total number of parameters

- Train the model using the iNaturalist dataset with 80:20 split for train and validation.
- Use W&B sweeps to tune hyperparameters like:
  - Number of filters
  - Filter organization
  - Activation function (ReLU, GELU, etc.)
  - Dropout and batch normalization
- Plot:
  - Accuracy vs creation time
  - Parallel coordinates
  - Correlation summary
- Write observations and use best model to predict test data.
- Visualize filters and guided backpropagation (optional).
- Report accuracy and include 10x3 image grid of test predictions.

---

### Part B: Fine-tuning a Pre-trained Model

- Choose a pre-trained model from torchvision (VGG, ResNet, Inception, ViT, etc.)
- Load ImageNet weights, modify input dimensions and the final output layer (from 1000 to 10 classes)
- Try at least 3 fine-tuning strategies like:
  - Freezing all layers except last
  - Unfreezing top-k layers
  - Full fine-tuning with a low learning rate
- Compare training-from-scratch and fine-tuned results
- Write insightful observations about performance, generalization, and convergence
- Report test accuracy and submit GitHub link

---

## Project Description

This project focuses on:
- Building a CNN from scratch (Part A)
- Fine-tuning the **VGG16** model (Part B)
- Classifying images into 10 classes using a subset of iNaturalist
- Using **Weights & Biases (W&B)** for experiment tracking and hyperparameter tuning

---

## Installation

Install dependencies using:

```bash
pip install -r requirements.txt

