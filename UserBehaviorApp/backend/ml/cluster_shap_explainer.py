"""SHAP-based explainability for clustering.

This module builds an interpretable surrogate model:

    cluster_label (target) <- X features (inputs)

Then computes SHAP values on that model to explain:
- Why a user is assigned to a particular cluster
- Which features push users toward high-spend / anomaly-like clusters

Important:
- SHAP is computed for the surrogate *classifier*, not directly for the clustering algorithm.
- The surrounding pipeline must provide:
  - X: feature dataframe used for clustering
  - cluster_labels: predicted cluster ids

Outputs:
- shap_values suitable for summary plots
- feature importance dataframe

Dependencies:
- shap
- xgboost or sklearn model

If SHAP is not installed, functions will raise a clear error.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _require_shap():
    try:
        import shap  # noqa: F401
    except Exception as e:
        raise ImportError(
            "SHAP is not installed. Install with: pip install shap"
        ) from e


def _build_classifier(use_xgb: bool = True, n_estimators: int = 300, random_state: int = 42):
    """Create a classifier for predicting cluster labels.

    Prefer XGBoost for strong explanations; fallback to LightGBM/XGB missing is left to user.
    """
    if use_xgb:
        try:
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=n_estimators,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
                n_jobs=-1,
                reg_lambda=1.0,
                objective="multi:softprob",
                eval_metric="mlogloss",
            )
        except Exception:
            # Fallback to sklearn without changing call sites.
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(
                n_estimators=400,
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced_subsample",
            )

    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=400,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )


def train_cluster_surrogate(
    X: pd.DataFrame,
    cluster_labels: np.ndarray,
    use_xgb: bool = True,
    model_params: Optional[Dict[str, Any]] = None,
):
    """Train a classifier to predict cluster labels from X."""

    if model_params is None:
        model_params = {}

    model = _build_classifier(use_xgb=use_xgb, **model_params)
    model.fit(X, cluster_labels)
    return model


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    nsamples: Optional[int] = 500,
):
    """Compute SHAP values for a trained model."""

    _require_shap()
    import shap

    # SHAP explainer works best when model is tree-based; for RF, KernelExplainer is heavy.
    # shap.Explainer will pick the right one when possible.
    if nsamples is not None and len(X) > nsamples:
        X_explain = X.sample(nsamples, random_state=42)
    else:
        X_explain = X

    explainer = shap.Explainer(model, X_explain)
    shap_values = explainer(X_explain)

    return shap_values, X_explain


def explain_features_for_cluster(
    shap_values: Any,
    X_explain: pd.DataFrame,
    cluster_labels_explain: np.ndarray,
    cluster_id: int,
    top_k: int = 10,
):
    """Return top-k features driving membership for one cluster.

    For multi-class SHAP outputs, shap_values.values is typically:
      shape = (n_samples, n_features, n_classes)

    If shap_values returns a different shape, this function makes a best effort.
    """

    values = shap_values.values

    # Handle multi-class
    # Common shape: (N, F, C)
    if values.ndim == 3:
        cluster_axis = 2
        # Mean absolute SHAP for samples belonging to cluster_id
        mask = cluster_labels_explain == cluster_id
        if mask.sum() == 0:
            return pd.DataFrame(columns=["feature", "mean_abs_shap"])

        abs_vals = np.abs(values[mask, :, cluster_id])
        mean_abs = abs_vals.mean(axis=0)
    else:
        # Binary or single output
        mask = cluster_labels_explain == cluster_id
        if mask.sum() == 0:
            return pd.DataFrame(columns=["feature", "mean_abs_shap"])
        mean_abs = np.abs(values[mask]).mean(axis=0)

    feature_names = list(X_explain.columns)
    out = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    out = out.sort_values("mean_abs_shap", ascending=False).head(top_k)
    return out


def build_shap_summary_dataframe(shap_values: Any, X_explain: pd.DataFrame):
    """Convenience: summary importance table (mean absolute SHAP per feature)."""
    values = shap_values.values
    if values.ndim == 3:
        # average across classes
        mean_abs = np.abs(values).mean(axis=(0, 2))
    else:
        mean_abs = np.abs(values).mean(axis=0)

    feature_names = list(X_explain.columns)
    return pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )

