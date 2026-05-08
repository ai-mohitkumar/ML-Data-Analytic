
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from io import StringIO
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.ensemble import IsolationForest

app = FastAPI(title="User Behavior Analytics API")

# Allow React frontend running on localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "User Behavior Analytics API is running"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Read uploaded CSV
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode("utf-8")))

    if df.empty:
        return {
            "message": "CSV is empty",
            "rows": 0,
            "columns": [],
            "preview": [],
            "clusters": [],
            "metrics": {},
            "patterns": [],
            "recommendations": [],
            "algorithm": {"selected": "N/A"},
            "anomalies": {"count": 0, "percentage": 0},
        }

    # Keep only numeric columns for ML analysis
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] == 0:
        return {
            "message": "No numeric columns found",
            "rows": len(df),
            "columns": list(df.columns),
            "preview": df.head().to_dict(orient="records"),
            "clusters": [],
            "metrics": {},
            "patterns": [],
            "recommendations": [],
            "algorithm": {"selected": "N/A"},
            "anomalies": {"count": 0, "percentage": 0},
        }

    # Fill missing values
    numeric_df = numeric_df.fillna(numeric_df.mean())

    # Scale data
    scaler = StandardScaler()
    X = scaler.fit_transform(numeric_df)

    # Choose number of clusters
    n_samples = len(numeric_df)
    k = 3 if n_samples >= 3 else 2
    k = min(k, n_samples)

    if n_samples < 2:
        labels = np.zeros(n_samples, dtype=int)
        clusters = [{
            "id": 0,
            "label": "Cluster 0",
            "size": int(n_samples),
            "percentage": 100.0,
            "avg_spending": 0.0,
            "avg_purchases": 0.0,
        }]
        metrics = {
            "silhouette": 0.0,
            "davies_bouldin": 0.0,
            "calinski_harabasz": 0.0,
            "icso_score": 0.0,
        }
    else:
        # KMeans clustering
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)

        # Build cluster summaries
        clusters = []
        spending_col = "spending_score" if "spending_score" in numeric_df.columns else numeric_df.columns[0]
        purchase_col = (
            "purchase_frequency"
            if "purchase_frequency" in numeric_df.columns
            else numeric_df.columns[min(1, len(numeric_df.columns) - 1)]
        )

        for cluster_id in range(k):
            cluster_rows = numeric_df[labels == cluster_id]
            size = len(cluster_rows)
            percentage = (size / n_samples) * 100 if n_samples else 0

            clusters.append({
                "id": int(cluster_id),
                "label": f"Cluster {cluster_id}",
                "size": int(size),
                "percentage": float(percentage),
                "avg_spending": float(cluster_rows[spending_col].mean()) if size else 0.0,
                "avg_purchases": float(cluster_rows[purchase_col].mean()) if size else 0.0,
            })

        # Metrics (only valid when at least 2 clusters are present)
        if len(set(labels)) > 1:
            silhouette = float(silhouette_score(X, labels))
            davies_bouldin = float(davies_bouldin_score(X, labels))
            calinski_harabasz = float(calinski_harabasz_score(X, labels))
        else:
            silhouette = 0.0
            davies_bouldin = 0.0
            calinski_harabasz = 0.0

        # Simple normalized composite score (ICSO)
        icso_score = max(
            0.0,
            min(
                1.0,
                (max(silhouette, 0.0) + (1.0 / (1.0 + max(davies_bouldin, 0.0)))) / 2.0,
            ),
        )

        metrics = {
            "silhouette": silhouette,
            "davies_bouldin": davies_bouldin,
            "calinski_harabasz": calinski_harabasz,
            "icso_score": icso_score,
        }

    # Anomaly detection
    if n_samples >= 5:
        iso = IsolationForest(contamination=0.1, random_state=42)
        anomaly_flags = iso.fit_predict(X)
        anomaly_count = int((anomaly_flags == -1).sum())
    else:
        anomaly_count = 0

    anomaly_percentage = (anomaly_count / n_samples) * 100 if n_samples else 0

    # Example recommendations
    recommendations = [
        {"product": "Premium Membership", "confidence": 0.92},
        {"product": "Discount Coupons", "confidence": 0.87},
        {"product": "Loyalty Rewards", "confidence": 0.81},
    ]

    # Example pattern mining placeholders
    patterns = [
        {"pattern": "High income users show higher spending scores"},
        {"pattern": "Frequent purchasers cluster together"},
    ]

    return {
        "message": "Analysis complete",
        "rows": int(len(df)),
        "columns": list(df.columns),
        "preview": df.head().to_dict(orient="records"),
        "clusters": clusters,
        "metrics": metrics,
        "patterns": patterns,
        "recommendations": recommendations,
        "algorithm": {"selected": "KMeans"},
        "anomalies": {
            "count": anomaly_count,
            "percentage": float(anomaly_percentage),
        },
    }
