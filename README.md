![Picture1](https://github.com/user-attachments/assets/fe14948d-b3c0-4f88-9b25-e7d94de24739)

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Seismic Ground Motion Modelling with Conditional Variational Autoencoder (CVAE)</title>
</head>
<body>
    <h1>Enhanced Seismic Ground Motion Modelling with Conditional Variational Autoencoder (CVAE)</h1>
    <p>This repository contains code for a Conditional Variational Autoencoder (CVAE) model designed for predicting seismic ground motion parameters. The model predicts 24 seismic response variables based on 7 input features. The training process involves hyperparameter optimization using grid search and cross-validation, followed by a final training phase with the selected best hyperparameters.</p>

    <h2>Table of Contents</h2>
    <ul>
        <li><a href="#project-overview">Project Overview</a></li>
        <li><a href="#requirements">Requirements</a></li>
        <li><a href="#setup-and-installation">Setup and Installation</a></li>
        <li><a href="#usage">Usage</a></li>
        <li><a href="#model-overview">Model Overview</a></li>
        <li><a href="#training">Training</a></li>
        <li><a href="#hyperparameter-tuning">Hyperparameter Tuning</a></li>
        <li><a href="#saving-models">Saving Models</a></li>
        <li><a href="#license">License</a></li>
    </ul>

    <h2 id="project-overview">Project Overview</h2>
    <p>This project aims to enhance seismic ground motion modelling by using a Conditional Variational Autoencoder (CVAE) to predict seismic response variables based on input features. It includes a mapping network that transforms the 7 input features into a latent space used by the decoder. The project also optimizes hyperparameters via grid search and cross-validation to find the best model configuration.</p>

    <h3>Key Features</h3>
    <ul>
        <li><strong>CVAE Architecture</strong>: The model consists of an encoder-decoder structure where the encoder compresses the input data into a latent space, and the decoder reconstructs the seismic response variables.</li>
        <li><strong>Mapping Network</strong>: A separate network that maps the 7 input features to the latent space used by the decoder for generating predictions.</li>
        <li><strong>Hyperparameter Tuning</strong>: Grid search to optimize hyperparameters like hidden layer sizes, learning rate, and dropout rate.</li>
    </ul>

    <h2 id="requirements">Requirements</h2>
    <p>To run this project, the following libraries are required:</p>
    <ul>
        <li>Python 3.x</li>
        <li>TensorFlow 2.x</li>
        <li>NumPy</li>
        <li>Pandas</li>
        <li>scikit-learn</li>
    </ul>
    <p>To install the dependencies, create a virtual environment and install them using the following command:</p>
    <pre><code>pip install -r requirements.txt</code></pre>

    <h3>requirements.txt</h3>
    <pre><code>
tensorflow==2.8.0
numpy==1.21.0
pandas==1.3.0
scikit-learn==0.24.2
    </code></pre>

    <h2 id="setup-and-installation">Setup and Installation</h2>
    <p>Follow these steps to set up the repository:</p>
    <ol>
        <li>Clone the repository to your local machine:
            <pre><code>git clone https://github.com/thunder-volt/nga_west2_cvae.git</code></pre>
        </li>
        <li>Navigate to the project directory:
            <pre><code>cd nga_west2_cvae</code></pre>
        </li>
        <li>Install the required libraries:
            <pre><code>pip install -r requirements.txt</code></pre>
        </li>
        <li>Place your dataset (seismic data) in CSV format with the following columns:
            <ul>
                <li>Input features: `['eqm', 'ftype', 'hyp', 'dist', 'log_dist', 'log_vs30', 'dir']`</li>
                <li>Output variables (24 seismic response variables): Columns like `['pga', 'T0.010S', 'T0.020S', ..., 'T4.000S']`</li>
            </ul>
        </li>
        <li>Modify the script to load your dataset (replace `df = pd.read_csv('your_data.csv')` with your dataset loading logic).</li>
    </ol>

    <h2 id="usage">Usage</h2>
    <p>To run the model, execute the Python file:</p>
    <pre><code>python nga_west2_cvae.py</code></pre>
    <p>The script will:</p>
    <ul>
        <li>Load and preprocess the dataset (log transform and MinMax scale the inputs and outputs).</li>
        <li>Perform grid search for hyperparameter optimization using 5-fold cross-validation.</li>
        <li>Train the final model using the best hyperparameters.</li>
        <li>Save the encoder, decoder, and final model as `.h5` files for later use.</li>
    </ul>

    <h3>Example to Use the Final Model for Prediction</h3>
    <p>Once the final model is saved as `final_model.h5`, you can load it and use it for making predictions:</p>
    <pre><code>
from tensorflow.keras.models import load_model
import numpy as np

# Load the final model
final_model = load_model('final_model.h5')

# Example input data for prediction (7 input features)
input_data = np.array([[0.5, 1.0, 2.0, 0.8, 1.5, 0.7, 2.5]])

# Predict the output (24 seismic variables)
predictions = final_model.predict(input_data)
print(predictions)
    </code></pre>

    <h2 id="model-overview">Model Overview</h2>
    <p>The model architecture consists of two main components:</p>
    <ul>
        <li><strong>Encoder</strong>: Encodes the seismic response variables into a latent space with a lower-dimensional representation.</li>
        <li><strong>Decoder</strong>: Reconstructs the seismic response variables from the latent space representation.</li>
    </ul>
    <p>A separate mapping network is used to convert the 7 input features into the latent space before passing them to the decoder for output generation.</p>

    <h2 id="training">Training</h2>
    <p>The training process involves the following key steps:</p>
    <ol>
        <li><strong>Data Preprocessing</strong>: The output variables are log-transformed for stability, and both input and output data are MinMax scaled to the range (-1, 1).</li>
        <li><strong>Hyperparameter Optimization</strong>: Grid search is conducted over hidden layer sizes, dropout rates, and learning rates. The best hyperparameters are selected based on the validation loss from cross-validation.</li>
        <li><strong>Final Training</strong>: After identifying the optimal hyperparameters, the model is trained on the full dataset.</li>
    </ol>

    <h2 id="hyperparameter-tuning">Hyperparameter Tuning</h2>
    <p>During the grid search, the following hyperparameters are tested:</p>
    <ul>
        <li><strong>Hidden Layer Sizes</strong>: Options like [16, 12, 8] and [12, 9, 6]</li>
        <li><strong>Dropout Rate</strong>: Options like [0.2, 0.3]</li>
        <li><strong>Learning Rate</strong>: Options like [0.0001, 0.0002]</li>
    </ul>
    <p>The script will print the validation loss for each combination of hyperparameters. The best configuration will then be used for final training.</p>

    <h2 id="saving-models">Saving Models</h2>
    <p>After training, the encoder, decoder, and final model are saved as `.h5` files:</p>
    <pre><code>
encoder.save("encoder_model.h5")
decoder.save("decoder_model.h5")
final_model.save("final_model.h5")
    </code></pre>
    <p>These files can be reloaded later for inference.</p>

    <h2 id="license">License</h2>
    <p>This project is licensed under the MIT License. See the LICENSE file for more details.</p>
</body>
</html>

