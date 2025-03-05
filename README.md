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
```

### requirements.txt
```txt
tensorflow==2.8.0
numpy==1.21.0
pandas==1.3.0
scikit-learn==0.24.2
```

## Setup and Installation
Follow these steps to set up the repository:

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/thunder-volt/CVAE.git
   ```

2. Navigate to the project directory:
   ```bash
   cd nga_west2_cvae
   ```

3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. Place your dataset (seismic data) in CSV format with the following columns:
   - Input features: `['eqm', 'ftype', 'hyp', 'dist', 'log_dist', 'log_vs30', 'dir']`
   - eqm: Earthquake Magnitude; ftype: Faulting Mechanism, hyp: Hypocentral Depth; dist: Joyner Boore Distance; log_dist: log(dist); log_vs30: logarithm of shear wave velocity of top 30 meter of soil; dir: direction of ground motion record
   - Output variables (24 seismic response variables): Columns like `['pga', 'T0.010S', 'T0.020S', ..., 'T4.000S']`

5. Modify the script to load your dataset (replace `df = pd.read_csv('your_data.csv')` with your dataset loading logic).

## Usage
To run the model, execute the Python file:

```bash
python nga_west2_cvae.py
```

The script will:
- Load and preprocess the dataset (log transform and MinMax scale the inputs and outputs).
- Perform grid search for hyperparameter optimization using 5-fold cross-validation.
- Train the final model using the best hyperparameters.
- Save the encoder, decoder, and final model as `.h5` files for later use.

### Example to Use the Final Model for Prediction
Once the final model is saved as `final_model.h5`, you can load it and use it for making predictions:

```python
from tensorflow.keras.models import load_model
import numpy as np

# Load the final model
final_model = load_model('final_model.h5')

# Example input data for prediction (7 input features)
input_data = np.array([[0.5, 1.0, 2.0, 0.8, 1.5, 0.7, 2.5]])

# Predict the output (24 seismic variables)
predictions = final_model.predict(input_data)
print(predictions)
```

## Model Overview
The model architecture consists of two main components:
- **Encoder**: Encodes the seismic response variables into a latent space with a lower-dimensional representation.
- **Decoder**: Reconstructs the seismic response variables from the latent space representation.

A separate mapping network is used to convert the 7 input features into the latent space before passing them to the decoder for output generation.

## Training
The training process involves the following key steps:
1. **Data Preprocessing**: The output variables are log-transformed for stability, and both input and output data are MinMax scaled to the range (-1, 1).
2. **Hyperparameter Optimization**: Grid search is conducted over hidden layer sizes, dropout rates, and learning rates. The best hyperparameters are selected based on the validation loss from cross-validation.
3. **Final Training**: After identifying the optimal hyperparameters, the model is trained on the full dataset.

## Hyperparameter Tuning
During the grid search, the following hyperparameters are tested:
- **Hidden Layer Sizes**: Options like [16, 12, 8] and [12, 9, 6]
- **Dropout Rate**: Options like [0.2, 0.3]
- **Learning Rate**: Options like [0.0001, 0.0002]

The script will print the validation loss for each combination of hyperparameters. The best configuration will then be used for final training.

## Saving Models
After training, the encoder, decoder, and final model are saved as `.h5` files:

```python
encoder.save("encoder_model.h5")
decoder.save("decoder_model.h5")
final_model.save("final_model.h5")
```

These files can be reloaded later for inference.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
```

### Key Sections in the README:
1. **Project Overview**: Summarizes the purpose and features of the project.
2. **Requirements**: Lists the necessary Python libraries and provides installation instructions.
3. **Setup and Installation**: Step-by-step guide for setting up the project environment.
4. **Usage**: Instructions for running the code and examples of how to use the model for predictions.
5. **Model Overview**: Explains the architecture of the model.
6. **Training**: Describes the steps involved in training the model.
7. **Hyperparameter Tuning**: Details the grid search process for hyperparameter optimization.
8. **Saving Models**: Explains how the model is saved for future use.
9. **License**: Includes licensing information.

You can copy this markdown and paste it into the `README.md` file in your repository. Let me know if you need any adjustments!

