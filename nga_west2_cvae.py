# -*- coding: utf-8 -*-
"""NGA_West2_CVAE.py

# Data Loading
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import randn
import os
import tensorflow as tf
from keras.layers import Concatenate
from tensorflow.keras.optimizers import Adam

from google.colab import drive

drive.mount('/content/drive')

df = pd.read_csv('path_to_data')
df = df.iloc[:,2:]
df.head(1)

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Dense, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

# -------------------------------
# 1. Data Preprocessing
# -------------------------------


# Define output columns (24 outputs)
outputs_columns = [
    'pga', 'T0.010S', 'T0.020S', 'T0.030S', 'T0.040S', 'T0.050S', 'T0.060S',
    'T0.070S', 'T0.080S', 'T0.090S', 'T0.150S', 'T0.200S', 'T0.300S',
    'T0.500S', 'T0.600S', 'T0.700S', 'T0.800S', 'T0.900S', 'T1.000S',
    'T1.200S', 'T1.500S', 'T2.000S', 'T3.000S', 'T4.000S'
]

# Log-transform the outputs for numerical stability
df[outputs_columns] = np.log1p(df[outputs_columns])

# Scale outputs to (-1, 1)
scaler_y = MinMaxScaler(feature_range=(-1, 1))
outputs = scaler_y.fit_transform(df[outputs_columns])

# Define and scale the 7 input features
input_columns = ['eqm', 'ftype', 'hyp', 'dist', 'log_dist', 'log_vs30', 'dir']
scaler_x = MinMaxScaler(feature_range=(-1, 1))
inputs = scaler_x.fit_transform(df[input_columns])

# Check the shapes of the inputs and outputs
print("Input shape:", inputs.shape)
print("Output shape:", outputs.shape)

# -------------------------------
# 2. Define Sampling Function and CVAE Class
# -------------------------------
def mse_loss(y_true, y_pred):
    # you can define your own loss function here (MSE/ MAE/ Combination of them etc...)
    return 0 # DO NOT FORGET TO UPDATE THIS PART AT "RETURN" 

# Sampling function for the reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Define a CVAE model by subclassing tf.keras.Model and overriding train_step
class CVAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(CVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            # Forward pass: encode then decode
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            # Compute reconstruction loss (sum of squared errors)
            reconstruction_loss = tf.reduce_sum(tf.square(data - reconstruction), axis=-1)
            # Compute KL divergence loss
            kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
            total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": total_loss}

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

# -------------------------------
# 3. Build CVAE (Encoder & Decoder) via a Function
# -------------------------------

def build_cvae(hl1, hl2, dropout_rate, learning_rate):
    latent_dim = 3  # Dimension of the latent space

    # Build the encoder network
    encoder_inputs = Input(shape=(24,), name='encoder_input')
    x = Dense(hl1, activation='relu')(encoder_inputs)
    x = Dropout(dropout_rate)(x)
    x = Dense(hl2, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z = Lambda(sampling, name='z')([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

    # Build the decoder network (symmetric to the encoder)
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x_dec = Dense(hl2, activation='relu')(latent_inputs)
    x_dec = Dense(hl1, activation='relu')(x_dec)
    decoder_outputs = Dense(24, activation='linear')(x_dec)
    decoder = Model(latent_inputs, decoder_outputs, name='decoder')

    # Build and compile the CVAE using the custom train_step
    cvae = CVAE(encoder, decoder)
    cvae.compile(optimizer=Adam(learning_rate=learning_rate), loss=mse_loss)  # Ensure loss is None
    return encoder, decoder, cvae

# -------------------------------
# 4. Grid Search with 5-Fold Cross Validation
# -------------------------------

hl1_options = [16, 12, 8]         # Options for hidden layer 1
hl2_options = [12, 9, 6]          # Options for hidden layer 2
learning_rates = [0.0001, 0.0002]  # Learning rate options
dropout_rates = [0.2, 0.3]         # Dropout rate options

num_folds = 5
epochs = 50      # Adjust epochs as needed
batch_size = 32

X_cvae = outputs

kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
best_val_loss = np.inf
best_params = None

print("Starting grid search for CVAE hyperparameters...")
for hl1 in hl1_options:
    for hl2 in hl2_options:
        for lr in learning_rates:
            for dr in dropout_rates:
                fold_val_losses = []
                for train_index, val_index in kf.split(X_cvae):
                    X_train, X_val = X_cvae[train_index], X_cvae[val_index]
                    print(f"Training shape: {X_train.shape}, Validation shape: {X_val.shape}")
                    encoder_tmp, decoder_tmp, cvae_tmp = build_cvae(hl1, hl2, dr, lr)
                    history = cvae_tmp.fit(X_train,
                                           epochs=epochs,
                                           batch_size=batch_size,
                                           validation_data=(X_val, X_val),  # Ensuring val_data is passed correctly
                                           verbose=0)
                    fold_val_losses.append(history.history['loss'][-1])

                avg_val_loss = np.mean(fold_val_losses)
                print(f"hl1: {hl1}, hl2: {hl2}, lr: {lr}, dropout: {dr}, avg_val_loss: {avg_val_loss:.4f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_params = (hl1, hl2, lr, dr)

print("\nBest hyperparameters found:", best_params, "with validation loss:", best_val_loss)

# Retrain CVAE on the full dataset using the best hyperparameters
best_hl1, best_hl2, best_lr, best_dr = best_params
encoder, decoder, cvae = build_cvae(best_hl1, best_hl2, best_dr, best_lr)
print("\nTraining final CVAE model on the full dataset...")
cvae.fit(X_cvae, epochs=epochs, batch_size=batch_size, verbose=1)

# Save the trained encoder and decoder models
encoder.save("encoder_model.h5")
decoder.save("decoder_model.h5")
print("Encoder and decoder models saved.")

# -------------------------------
# 5. Build the Mapping Network & Final Model
# -------------------------------

def build_mapping_network():
    latent_dim = 3
    mapping_input = Input(shape=(7,), name='mapping_input')
    x = Dense(7, activation='relu')(mapping_input)
    x = Dense(4, activation='relu')(x)
    latent_output = Dense(latent_dim, activation='linear', name='latent_output')(x)
    return Model(mapping_input, latent_output, name='mapping_network')

mapping_network = build_mapping_network()

# Build final model: mapping network + frozen decoder from the CVAE
final_input = Input(shape=(7,), name='final_input')
latent_pred = mapping_network(final_input)
decoder.trainable = False  # Freeze decoder weights
final_output = decoder(latent_pred)
final_model = Model(final_input, final_output, name='final_model')

# Compile final model (predicting 24 outputs from 7 inputs)
final_model.compile(optimizer=Adam(learning_rate=best_lr), loss='mse')

print("\nTraining final mapping model (mapping network + decoder)...")
final_model.fit(inputs, outputs, epochs=epochs, batch_size=batch_size, verbose=1)

# Save the final model
final_model.save("final_model.h5")
print("Final model saved as 'final_model.h5'.")

