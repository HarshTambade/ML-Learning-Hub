"""Autoencoders for Dimensionality Reduction"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

print("Autoencoders for Dimensionality Reduction")
print("="*60)

# Load data
digits = load_digits()
X_digits = digits.data / 255.0
y_digits = digits.target

iris = load_iris()
X_iris = StandardScaler().fit_transform(iris.data)
y_iris = iris.target

print(f"\nDatasets loaded:")
print(f"  Digits: {X_digits.shape}")
print(f"  Iris: {X_iris.shape}")

# 1. Basic Autoencoder for Dimensionality Reduction
print("\n1. Basic Autoencoder")
print("-" * 60)

encoding_dim = 32
input_dim = X_digits.shape[1]

# Encoder
input_img = keras.Input(shape=(input_dim,))
encoded = layers.Dense(64, activation='relu')(input_img)
encoded = layers.Dense(32, activation='relu')(encoded)
encoded = layers.Dense(encoding_dim, activation='relu')(encoded)

# Decoder
decoded = layers.Dense(32, activation='relu')(encoded)
decoded = layers.Dense(64, activation='relu')(decoded)
decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)

# Autoencoder
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

print(f"Autoencoder architecture:")
print(f"  Input dimension: {input_dim}")
print(f"  Encoding dimension: {encoding_dim}")
print(f"  Compression ratio: {input_dim / encoding_dim:.2f}x")

# Train autoencoder
X_train_dig, X_test_dig = train_test_split(X_digits, test_size=0.2, random_state=42)

history = autoencoder.fit(
    X_train_dig, X_train_dig,
    epochs=50,
    batch_size=256,
    validation_data=(X_test_dig, X_test_dig),
    verbose=0
)

print(f"Training complete!")
print(f"  Final training loss: {history.history['loss'][-1]:.6f}")
print(f"  Final validation loss: {history.history['val_loss'][-1]:.6f}")

# Test reconstruction
X_test_decoded = autoencoder.predict(X_test_dig, verbose=0)
mse = mean_squared_error(X_test_dig, X_test_decoded)
print(f"  Test MSE: {mse:.6f}")

# 2. Extract Encoder for Dimensionality Reduction
print("\n2. Using Encoder for Dimensionality Reduction")
print("-" * 60)

encoder = Model(input_img, encoded)
X_digits_encoded = encoder.predict(X_digits, verbose=0)

print(f"Original shape: {X_digits.shape}")
print(f"Encoded shape: {X_digits_encoded.shape}")
print(f"Reduction: {X_digits.shape[1]} -> {X_digits_encoded.shape[1]} dimensions")
print(f"Size reduction: {100 * (1 - X_digits_encoded.shape[1] / X_digits.shape[1]):.1f}%")

# 3. Variational Autoencoder (VAE) - More Advanced
print("\n3. Variational Autoencoder (VAE)")
print("-" * 60)

latent_dim = 20

# Sampling layer
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# VAE Encoder
inputs = keras.Input(shape=(input_dim,))
x = layers.Dense(64, activation="relu")(inputs)
x = layers.Dense(32, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder_vae = Model(inputs, [z_mean, z_log_var, z], name="encoder")

print("VAE Encoder:")
encoder_vae.summary()

# VAE Decoder
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(32, activation="relu")(latent_inputs)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(input_dim, activation="sigmoid")(x)
decoder_vae = Model(latent_inputs, outputs, name="decoder")

print("\nVAE Decoder:")
decoder_vae.summary()

# VAE Model
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=1,
                )
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

vae = VAE(encoder_vae, decoder_vae)
vae.compile(optimizer=Adam(learning_rate=1e-3))

history_vae = vae.fit(X_train_dig, epochs=50, batch_size=256, verbose=0)

print(f"\nVAE Training complete!")
print(f"  Final loss: {history_vae.history['loss'][-1]:.6f}")
print(f"  Reconstruction loss: {history_vae.history['reconstruction_loss'][-1]:.6f}")
print(f"  KL loss: {history_vae.history['kl_loss'][-1]:.6f}")

# 4. Denoising Autoencoder
print("\n4. Denoising Autoencoder")
print("-" * 60)

X_train_noisy = X_train_dig + 0.2 * np.random.randn(*X_train_dig.shape)
X_test_noisy = X_test_dig + 0.2 * np.random.randn(*X_test_dig.shape)

X_train_noisy = np.clip(X_train_noisy, 0, 1)
X_test_noisy = np.clip(X_test_noisy, 0, 1)

# Denoising Autoencoder model
input_noisy = keras.Input(shape=(input_dim,))
encoded_den = layers.Dense(64, activation='relu')(input_noisy)
encoded_den = layers.Dense(32, activation='relu')(encoded_den)
encoded_den = layers.Dense(encoding_dim, activation='relu')(encoded_den)

decoded_den = layers.Dense(32, activation='relu')(encoded_den)
decoded_den = layers.Dense(64, activation='relu')(decoded_den)
decoded_den = layers.Dense(input_dim, activation='sigmoid')(decoded_den)

autoencoder_denoising = Model(input_noisy, decoded_den)
autoencoder_denoising.compile(optimizer='adam', loss='mse')

history_denoise = autoencoder_denoising.fit(
    X_train_noisy, X_train_dig,
    epochs=50,
    batch_size=256,
    validation_data=(X_test_noisy, X_test_dig),
    verbose=0
)

print(f"Denoising Autoencoder trained!")
print(f"  Final training loss: {history_denoise.history['loss'][-1]:.6f}")
print(f"  Final validation loss: {history_denoise.history['val_loss'][-1]:.6f}")

# Test denoising
X_test_denoised = autoencoder_denoising.predict(X_test_noisy, verbose=0)
mse_noisy = mean_squared_error(X_test_noisy, X_test_dig)
mse_denoised = mean_squared_error(X_test_denoised, X_test_dig)

print(f"\n  Noisy vs Original MSE: {mse_noisy:.6f}")
print(f"  Denoised vs Original MSE: {mse_denoised:.6f}")
print(f"  Improvement: {100 * (mse_noisy - mse_denoised) / mse_noisy:.1f}%")

# 5. Visualization
print("\n5. Visualization")
print("-" * 60)

fig, axes = plt.subplots(3, 3, figsize=(15, 10))

# Training history
ax1 = plt.subplot(2, 3, 1)
ax1.plot(history.history['loss'], label='Training Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Autoencoder Training History')
ax1.legend()

# VAE training history
ax2 = plt.subplot(2, 3, 2)
ax2.plot(history_vae.history['loss'], label='Total Loss')
ax2.plot(history_vae.history['reconstruction_loss'], label='Reconstruction')
ax2.plot(history_vae.history['kl_loss'], label='KL Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('VAE Training History')
ax2.legend()

# Denoising history
ax3 = plt.subplot(2, 3, 3)
ax3.plot(history_denoise.history['loss'], label='Training Loss')
ax3.plot(history_denoise.history['val_loss'], label='Validation Loss')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss')
ax3.set_title('Denoising Autoencoder History')
ax3.legend()

# Original digits
ax4 = plt.subplot(2, 3, 4)
for i in range(9):
    ax = plt.subplot(3, 3, i+4)
    digit = X_test_dig[i].reshape(8, 8)
    ax.imshow(digit, cmap='gray')
    ax.set_title('Original')
    ax.axis('off')

plt.suptitle('Autoencoder Reconstruction Examples')
plt.tight_layout()
plt.savefig('autoencoder_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("Autoencoder analysis complete!")
print("Visualization saved as 'autoencoder_analysis.png'")
