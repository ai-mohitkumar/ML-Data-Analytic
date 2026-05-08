"""LSTM-based user behavior prediction (Phase 5).

This module provides an end-to-end workflow to predict a future numeric target
(e.g., spend, clicks, session_duration) from sequences of past behavior.

It is implemented to be generic and safe:
- Uses TensorFlow/Keras only when training/predicting is invoked.
- Includes helpers to build sequences from a pandas DataFrame.

Typical usage:

    from UserBehaviorApp.backend.ml.lstm_user_prediction import (
        build_lstm_model,
        make_sequences,
        train_lstm_on_dataframe,
    )

    model, metrics = train_lstm_on_dataframe(
        df, user_id_col='user_id', time_col='day',
        feature_cols=['spend', 'clicks'], target_col='spend',
        lookback=14, horizon=1,
    )

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class LSTMData:
    X: np.ndarray  # (samples, timesteps, features)
    y: np.ndarray  # (samples,)
    feature_cols: List[str]


def make_sequences(
    df: pd.DataFrame,
    *,
    user_id_col: str,
    time_col: str,
    feature_cols: List[str],
    target_col: str,
    lookback: int,
    horizon: int = 1,
    sort_ascending: bool = True,
) -> LSTMData:
    """Convert a long-format dataframe into LSTM supervised sequences.

    For each user:
      - sort by time
      - create windows of length `lookback` to predict target at t + horizon

    Returns:
        LSTMData with X and y.
    """

    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    for c in [user_id_col, time_col, target_col] + feature_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    X_list = []
    y_list = []

    # Ensure numeric
    work = df[[user_id_col, time_col] + feature_cols + [target_col]].copy()
    for c in feature_cols + [target_col]:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=feature_cols + [target_col])

    # Group per user
    for uid, g in work.groupby(user_id_col):
        g = g.sort_values(time_col, ascending=sort_ascending)
        arr_feat = g[feature_cols].values.astype(np.float32)
        arr_y = g[target_col].values.astype(np.float32)

        # Index for last window start
        # We need target at (start + lookback - 1) + horizon
        max_start = len(g) - lookback - horizon + 1
        if max_start <= 0:
            continue

        for start in range(max_start):
            end = start + lookback
            target_idx = end - 1 + horizon
            X_list.append(arr_feat[start:end])
            y_list.append(arr_y[target_idx])

    if not X_list:
        raise ValueError("No sequences could be created. Check lookback/horizon or data density.")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.float32)

    return LSTMData(X=X, y=y, feature_cols=feature_cols)


def build_lstm_model(
    *,
    timesteps: int,
    n_features: int,
    lstm_units: int = 64,
    dense_units: int = 32,
    learning_rate: float = 1e-3,
) -> Any:
    """Build and compile a simple LSTM regression model."""

    # Lazy imports so rest of repo doesn't require TF.
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=(timesteps, n_features)))
    model.add(Dropout(0.2))
    model.add(Dense(dense_units, activation="relu"))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model


def train_lstm_on_dataframe(
    df: pd.DataFrame,
    *,
    user_id_col: str,
    time_col: str,
    feature_cols: List[str],
    target_col: str,
    lookback: int = 14,
    horizon: int = 1,
    epochs: int = 20,
    batch_size: int = 32,
    validation_split: float = 0.2,
    lstm_units: int = 64,
    dense_units: int = 32,
    learning_rate: float = 1e-3,
    verbose: int = 1,
) -> Tuple[Any, Dict[str, float]]:
    """Train LSTM on sequences built from a dataframe."""

    data = make_sequences(
        df,
        user_id_col=user_id_col,
        time_col=time_col,
        feature_cols=feature_cols,
        target_col=target_col,
        lookback=lookback,
        horizon=horizon,
    )

    # Simple normalization per-feature (fit on all data; for production use train-only)
    X = data.X
    X_mean = X.mean(axis=(0, 1), keepdims=True)
    X_std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    Xn = (X - X_mean) / X_std

    y = data.y
    y_mean = y.mean()
    y_std = y.std() + 1e-8
    yn = (y - y_mean) / y_std

    model = build_lstm_model(
        timesteps=Xn.shape[1],
        n_features=Xn.shape[2],
        lstm_units=lstm_units,
        dense_units=dense_units,
        learning_rate=learning_rate,
    )

    history = model.fit(
        Xn,
        yn,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=verbose,
    )

    # Basic metric: final val loss if present
    metrics: Dict[str, float] = {}
    if hasattr(history, "history"):
        val_losses = history.history.get("val_loss")
        train_losses = history.history.get("loss")
        if val_losses:
            metrics["val_mse"] = float(val_losses[-1])
        if train_losses:
            metrics["train_mse"] = float(train_losses[-1])

    # Attach normalization params for downstream inference
    model._x_mean = X_mean
    model._x_std = X_std
    model._y_mean = y_mean
    model._y_std = y_std

    return model, metrics


def predict_next(
    model: Any,
    df: pd.DataFrame,
    *,
    user_id_col: str,
    time_col: str,
    feature_cols: List[str],
    lookback: int,
) -> pd.DataFrame:
    """Predict next-step target for each user based on latest lookback window.

    Assumes model was trained as a regression output (single numeric).

    Returns:
        DataFrame with user_id_col and prediction.
    """

    if not hasattr(model, "_x_mean"):
        raise ValueError("Model normalization parameters not found. Train using train_lstm_on_dataframe().")

    for c in [user_id_col, time_col] + feature_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    work = df[[user_id_col, time_col] + feature_cols].copy()
    for c in feature_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=feature_cols)

    preds = []
    for uid, g in work.groupby(user_id_col):
        g = g.sort_values(time_col, ascending=True)
        if len(g) < lookback:
            continue
        x_win = g[feature_cols].values.astype(np.float32)[-lookback:]
        x_win = x_win.reshape(1, lookback, len(feature_cols))

        # Normalize
        Xn = (x_win - model._x_mean) / model._x_std
        yn_pred = model.predict(Xn, verbose=0).reshape(-1)
        y_pred = yn_pred * model._y_std + model._y_mean

        preds.append({user_id_col: uid, "prediction": float(y_pred[0])})

    return pd.DataFrame(preds)

