"""Deep Learning Clustering Extension.

Implements Phase 1: Deep Learning Clustering via autoencoder latent features.

It is written as a separate orchestrator so the existing system can stay stable.
Later you can switch main/intelligent endpoints to use this module.

Key idea:
- scale features
- train autoencoder on scaled X
- embed => latent_features
- cluster latent_features with the same AutoML routine
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from UserBehaviorApp.backend.intelligent_engine import IntelligentAdaptiveClusteringEngine


def run_intelligent_analysis_with_deep_autoencoder(df, features, latent_dim: int = 16):
    """Run clustering on autoencoder latent space.

    This function assumes your clustering engine can operate on a numeric dataframe.
    For now we create a temporary dataframe of latent features and cluster those.
    """

    # Import lazily to avoid forcing TF for non-deep usage
    from UserBehaviorApp.backend.ml.deep_autoencoder import build_autoencoder

    X = df[features].select_dtypes(include=[np.number]).fillna(0.0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Build/Train autoencoder
    autoencoder, encoder = build_autoencoder(input_dim=X_scaled.shape[1])

    # Train (you can tune epochs/batch size later)
    autoencoder.fit(
        X_scaled,
        X_scaled,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0,
    )

    # Extract latent features
    latent_features = encoder.predict(X_scaled, verbose=0)

    # Create latent dataframe so IntelligentAdaptiveClusteringEngine can accept it
    latent_cols = [f"latent_{i}" for i in range(latent_features.shape[1])]
    df_latent = df.copy()
    for i, c in enumerate(latent_cols):
        df_latent[c] = latent_features[:, i]

    # Cluster in latent space
    deep_engine = IntelligentAdaptiveClusteringEngine(df_latent, latent_cols)
    results = deep_engine.run_full_analysis()

    return {
        "deep_autoencoder": True,
        "scaler": scaler,
        "latent_features": latent_features,
        "clustering_results": results,
    }

