"""Incremental clustering utilities (Phase 3).

Implements a streaming-friendly KMeans using MiniBatchKMeans.

Key method:
- partial_fit(new_data) updates cluster centroids incrementally.

This is used to make the system support "real-time" user updates.

Note: MiniBatchKMeans is incremental only for KMeans-style clustering.
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


def build_incremental_kmeans(
    n_clusters: int,
    feature_cols: List[str],
    random_state: int = 42,
    batch_size: int = 256,
    n_init: int = 3,
) -> MiniBatchKMeans:
    """Create an untrained MiniBatchKMeans model."""
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=batch_size,
        n_init=n_init,
    )


def incremental_partial_fit(
    model: MiniBatchKMeans,
    df_new: pd.DataFrame,
    feature_cols: List[str],
) -> MiniBatchKMeans:
    """Update clustering centroids with new user feature rows."""
    X_new = df_new[feature_cols].select_dtypes(include=[np.number]).fillna(0.0).values
    model.partial_fit(X_new)
    return model


def predict_clusters(
    model: MiniBatchKMeans,
    df: pd.DataFrame,
    feature_cols: List[str],
) -> np.ndarray:
    """Predict cluster ids for input rows."""
    X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0.0).values
    return model.predict(X)


def fit_and_incremental_train(
    df_initial: pd.DataFrame,
    df_stream: pd.DataFrame,
    feature_cols: List[str],
    n_clusters: int,
    random_state: int = 42,
) -> Tuple[MiniBatchKMeans, np.ndarray]:
    """Convenience method: fit on initial data, then update with stream once."""
    model = build_incremental_kmeans(
        n_clusters=n_clusters,
        feature_cols=feature_cols,
        random_state=random_state,
    )

    X0 = df_initial[feature_cols].select_dtypes(include=[np.number]).fillna(0.0).values
    model.fit(X0)

    incremental_partial_fit(model, df_stream, feature_cols=feature_cols)

    pred_stream = predict_clusters(model, df_stream, feature_cols=feature_cols)
    return model, pred_stream

