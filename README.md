![Picture1](https://github.com/user-attachments/assets/fe14948d-b3c0-4f88-9b25-e7d94de24739)

# Enhanced Seismic Ground Motion Modelling with Conditional Variational Autoencoder (CVAE)

This repository contains code for a Conditional Variational Autoencoder (CVAE) model designed for predicting seismic ground motion parameters. The model predicts 24 seismic response variables based on 7 input features. The training process involves hyperparameter optimization using grid search and cross-validation, followed by a final training phase with the selected best hyperparameters.

## Table of Contents
- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Model Overview](#model-overview)
- [Training](#training)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Saving Models](#saving-models)
- [License](#license)

## Project Overview
This project aims to enhance seismic ground motion modelling by using a Conditional Variational Autoencoder (CVAE) to predict seismic response variables based on input features. It includes a mapping network that transforms the 7 input features into a latent space used by the decoder. The project also optimizes hyperparameters via grid search and cross-validation to find the best model configuration.

### Key Features
- **CVAE Architecture**: The model consists of an encoder-decoder structure where the encoder compresses the input data into a latent space, and the decoder reconstructs the seismic response variables.
- **Mapping Network**: A separate network that maps the 7 input features to the latent space used by the decoder for generating predictions.
- **Hyperparameter Tuning**: Grid search to optimize hyperparameters like hidden layer sizes, learning rate, and dropout rate.

## Requirements
To run this project, the following libraries are required:
- Python 3.x
- TensorFlow 2.x
- NumPy
- Pandas
- scikit-learn

To install the dependencies, create a virtual environment and install them using the following command:

```bash
pip install -r requirements.txt

tensorflow==2.8.0
numpy==1.21.0
pandas==1.3.0
scikit-learn==0.24.2


