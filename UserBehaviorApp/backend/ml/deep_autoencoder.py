"""Deep Autoencoder for clustering latent feature extraction.

This module provides a lightweight autoencoder architecture that can be trained on
scaled numeric feature vectors, producing a lower-dimensional latent embedding.

Typical usage:

    autoencoder, encoder = build_autoencoder(input_dim)
    autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, validation_split=0.2)
    latent_features = encoder.predict(X_scaled)

Then cluster using latent_features instead of X_scaled.
"""

from __future__ import annotations

from typing import Tuple


def build_autoencoder(input_dim: int) -> Tuple["object", "object"]:
    """Builds a basic fully-connected autoencoder.

    Returns:
        autoencoder: compiled keras Model mapping input -> reconstructed input
        encoder: keras Model mapping input -> latent vector
    """

    # Import inside function so the rest of the project can run without TF installed.
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.optimizers import Adam

    input_layer = Input(shape=(input_dim,), name="input")

    # Encoder
    encoded = Dense(64, activation="relu", name="enc_dense_64")(input_layer)
    encoded = Dense(32, activation="relu", name="enc_dense_32")(encoded)
    latent = Dense(16, activation="relu", name="latent")(encoded)

    # Decoder
    decoded = Dense(32, activation="relu", name="dec_dense_32")(latent)
    decoded = Dense(64, activation="relu", name="dec_dense_64")(decoded)

    # NOTE: Using sigmoid assumes inputs are in [0, 1]. For general scaled inputs,
    # you may want to switch this to linear + mse.
    output = Dense(input_dim, activation="sigmoid", name="reconstruction")(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output, name="autoencoder")
    encoder = Model(inputs=input_layer, outputs=latent, name="encoder")

    autoencoder.compile(optimizer=Adam(0.001), loss="mse")

    return autoencoder, encoder

