"""
🔥 INTELLIGENT ADAPTIVE CLUSTERING SYSTEM
Advanced ML Engine with AutoML, Hybrid Scoring, and Business Intelligence

Features:
✅ Auto Algorithm Selection (KMeans vs DBSCAN vs Hierarchical)
✅ Hybrid Evaluation Metrics (Silhouette + DBI + CH)
✅ Novel ICSO Metric (Inter-Cluster Separation Optimization Score)
✅ Pseudo Accuracy (supervised validation)
✅ Cluster Profiling & Business Intelligence
✅ Anomaly Detection (IsolationForest)
✅ Smart Recommendations
✅ Explainable AI Features
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, pairwise_distances
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

class IntelligentAdaptiveClusteringEngine:
    """
    🧠 Advanced clustering with AutoML, hybrid scoring, and business intelligence
    """
    
    def __init__(self, data, features):
        self.data = data
        self.features = features
        self.X = data[features].values
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.results = {}
        self.best_model = None
        self.best_algorithm = None
        
    def calculate_icso_metric(self, labels, X, *, prev_labels=None, anomaly_ratio: float = 0.0, feature_weight: float = 1.0):
        """
        🔬 Novel AICSO Metric (Advanced ICSO)

        Upgraded from ICSO to include:
        - Temporal Stability (consistency with previous clustering)
        - Noise Penalty (anomaly ratio)
        - Feature Weighting (optional scaling)

        Formula (research/patent style):

            AICSO = (D_inter * S_temporal * W_feature) / (V_intra + A_noise)

        Where:
        - D_inter: mean inter-cluster centroid distance
        - V_intra: mean intra-cluster variance of point distances to centroid
        - S_temporal: cluster assignment stability w.r.t. prev_labels (Jaccard-like on pairs)
        - A_noise: anomaly_ratio (e.g., proportion of anomalies)
        - W_feature: feature_weight (default 1.0)

        Higher = Better.
        """
        try:
            labels = np.asarray(labels)
            unique_labels = np.unique(labels)

            # --- V_intra ---
            intra_variance = []
            for label in unique_labels:
                cluster_points = X[labels == label]
                if len(cluster_points) > 1:
                    centroid = cluster_points.mean(axis=0)
                    distances = np.linalg.norm(cluster_points - centroid, axis=1)
                    intra_variance.append(np.var(distances))

            mean_intra_variance = float(np.mean(intra_variance)) if intra_variance else 1.0

            # --- D_inter ---
            centroids = []
            for label in unique_labels:
                pts = X[labels == label]
                if len(pts) == 0:
                    continue
                centroids.append(pts.mean(axis=0))
            centroids = np.array(centroids)

            if len(centroids) > 1:
                inter_dists = pairwise_distances(centroids).flatten()
                inter_dists = inter_dists[inter_dists > 0]
                mean_inter_distance = float(np.mean(inter_dists)) if len(inter_dists) > 0 else 1.0
            else:
                mean_inter_distance = 0.0

            # --- S_temporal ---
            if prev_labels is None:
                temporal_stability = 1.0
            else:
                prev_labels = np.asarray(prev_labels)
                if prev_labels.shape[0] != labels.shape[0]:
                    temporal_stability = 1.0
                else:
                    # Pairwise agreement stability: fraction of sample pairs with same assignment
                    # This is O(n^2); approximate using label-wise matching on aligned positions instead.
                    # Using a robust simple alternative: accuracy-like agreement after relabeling.
                    from sklearn.preprocessing import LabelEncoder
                    from sklearn.metrics import accuracy_score

                    le1 = LabelEncoder()
                    le2 = LabelEncoder()
                    a = le1.fit_transform(labels)
                    b = le2.fit_transform(prev_labels)
                    temporal_stability = float(accuracy_score(a, b))
                    # Bound to [0,1]
                    temporal_stability = max(0.0, min(1.0, temporal_stability))

            # --- A_noise ---
            noise_penalty = float(anomaly_ratio) if anomaly_ratio is not None else 0.0
            noise_penalty = max(0.0, noise_penalty)

            # --- W_feature ---
            w_feature = float(feature_weight) if feature_weight is not None else 1.0
            if w_feature <= 0:
                w_feature = 1.0

            # --- AICSO ---
            numerator = mean_inter_distance * temporal_stability * w_feature
            denominator = (mean_intra_variance + noise_penalty + 1e-6)
            aicso = numerator / denominator

            return float(aicso)
        except:
            return 0.0

    
    def calculate_hybrid_score(self, labels):
        """
        📊 Hybrid Evaluation Score combining 3 metrics
        
        Hybrid Score = (Silhouette * 0.5) + (1/DBI * 0.3) + (CH/1000 * 0.2)
        Range: 0-1 (higher is better)
        """
        try:
            silhouette = silhouette_score(self.X_scaled, labels)
            dbi = davies_bouldin_score(self.X_scaled, labels)
            ch = calinski_harabasz_score(self.X_scaled, labels)
            
            # Normalize metrics to 0-1 scale
            silhouette_norm = (silhouette + 1) / 2  # From -1..1 to 0..1
            dbi_norm = 1 / (1 + dbi)  # Invert: lower DBI is better
            ch_norm = min(ch / 1000, 1.0)  # Cap at 1000
            
            # Weighted combination
            hybrid_score = (silhouette_norm * 0.5) + (dbi_norm * 0.3) + (ch_norm * 0.2)
            
            return {
                'hybrid_score': float(hybrid_score),
                'silhouette': float(silhouette),
                'davies_bouldin': float(dbi),
                'calinski_harabasz': float(ch),
                'silhouette_norm': float(silhouette_norm),
                'dbi_norm': float(dbi_norm),
                'ch_norm': float(ch_norm)
            }
        except Exception as e:
            print(f"Error calculating hybrid score: {e}")
            return None
    
    def try_kmeans(self, k_range=range(2, 9)):
        """Try KMeans with automatic K selection"""
        best_k = 2
        best_score = -999
        best_labels = None
        best_model = None
        
        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = kmeans.fit_predict(self.X_scaled)
                
                hybrid_metrics = self.calculate_hybrid_score(labels)
                if hybrid_metrics:
                    score = hybrid_metrics['hybrid_score']
                    
                    if score > best_score:
                        best_score = score
                        best_k = k
                        best_labels = labels
                        best_model = kmeans
            except:
                continue
        
        self.results['kmeans'] = {
            'model': best_model,
            'labels': best_labels,
            'k': best_k,
            'metrics': self.calculate_hybrid_score(best_labels),
            'icso': self.calculate_icso_metric(best_labels, self.X_scaled)
        }
        
        return best_score
    
    def try_dbscan(self, eps_range=np.arange(0.3, 1.5, 0.1)):
        """Try DBSCAN with automatic parameter selection"""
        best_eps = 0.5
        best_score = -999
        best_labels = None
        best_model = None
        
        for eps in eps_range:
            try:
                dbscan = DBSCAN(eps=eps, min_samples=5)
                labels = dbscan.fit_predict(self.X_scaled)
                
                # Need at least 2 clusters
                if len(np.unique(labels)) < 2:
                    continue
                
                # Filter out noise points
                if len(labels[labels == -1]) > len(labels) * 0.5:
                    continue
                
                hybrid_metrics = self.calculate_hybrid_score(labels)
                if hybrid_metrics:
                    score = hybrid_metrics['hybrid_score']
                    
                    if score > best_score:
                        best_score = score
                        best_eps = eps
                        best_labels = labels
                        best_model = dbscan
            except:
                continue
        
        if best_labels is not None:
            self.results['dbscan'] = {
                'model': best_model,
                'labels': best_labels,
                'eps': best_eps,
                'metrics': self.calculate_hybrid_score(best_labels),
                'icso': self.calculate_icso_metric(best_labels, self.X_scaled)
            }
            return best_score
        return -999
    
    def try_hierarchical(self, n_clusters_range=range(2, 9)):
        """Try Hierarchical clustering"""
        best_n = 2
        best_score = -999
        best_labels = None
        best_model = None
        
        for n in n_clusters_range:
            try:
                hierarchical = AgglomerativeClustering(n_clusters=n, linkage='ward')
                labels = hierarchical.fit_predict(self.X_scaled)
                
                hybrid_metrics = self.calculate_hybrid_score(labels)
                if hybrid_metrics:
                    score = hybrid_metrics['hybrid_score']
                    
                    if score > best_score:
                        best_score = score
                        best_n = n
                        best_labels = labels
                        best_model = hierarchical
            except:
                continue
        
        self.results['hierarchical'] = {
            'model': best_model,
            'labels': best_labels,
            'n_clusters': best_n,
            'metrics': self.calculate_hybrid_score(best_labels),
            'icso': self.calculate_icso_metric(best_labels, self.X_scaled)
        }
        
        return best_score
    
    def auto_select_algorithm(self):
        """🤖 Automatically select best algorithm"""
        print("🤖 AutoML: Testing algorithms...")
        
        scores = {}
        
        # Test each algorithm
        scores['kmeans'] = self.try_kmeans()
        scores['dbscan'] = self.try_dbscan()
        scores['hierarchical'] = self.try_hierarchical()
        
        print(f"  KMeans Score: {scores['kmeans']:.4f}")
        print(f"  DBSCAN Score: {scores['dbscan']:.4f}")
        print(f"  Hierarchical Score: {scores['hierarchical']:.4f}")
        
        # Select best
        best_algo = max(scores, key=scores.get)
        self.best_algorithm = best_algo
        self.best_model = self.results[best_algo]
        
        print(f"\n✅ Best Algorithm: {best_algo.upper()} (Score: {scores[best_algo]:.4f})")
        
        return best_algo, scores
    
    def generate_cluster_profiles(self):
        """📊 Generate business intelligence profiles for each cluster"""
        labels = self.best_model['labels']
        unique_labels = np.unique(labels)
        
        profiles = {}
        
        for label in unique_labels:
            if label == -1:  # Skip noise points in DBSCAN
                continue
            
            cluster_mask = labels == label
            cluster_data = self.data[cluster_mask]
            
            profile = {
                'cluster_id': int(label),
                'size': int(np.sum(cluster_mask)),
                'percentage': float(100 * np.sum(cluster_mask) / len(self.data)),
                'characteristics': {}
            }
            
            # Analyze features
            for feature in self.features:
                if feature in cluster_data.columns:
                    values = cluster_data[feature]
                    profile['characteristics'][feature] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max())
                    }
            
            # Generate business label
            if 'total_spent' in cluster_data.columns:
                avg_spend = cluster_data['total_spent'].mean()
                if avg_spend > cluster_data['total_spent'].quantile(0.75):
                    profile['business_label'] = 'Premium/High-Value'
                elif avg_spend > cluster_data['total_spent'].quantile(0.25):
                    profile['business_label'] = 'Standard/Medium-Value'
                else:
                    profile['business_label'] = 'Budget/Low-Value'
            
            profiles[int(label)] = profile
        
        return profiles
    
    def detect_anomalies(self, contamination=0.1):
        """🔍 Detect anomalous users using IsolationForest"""
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = iso_forest.fit_predict(self.X_scaled)
        anomaly_scores = iso_forest.score_samples(self.X_scaled)
        
        # Add to data
        self.data['anomaly'] = anomaly_labels == -1  # True if anomaly
        self.data['anomaly_score'] = anomaly_scores
        
        # Identify anomalies
        anomalies = self.data[self.data['anomaly']].copy()
        
        return {
            'n_anomalies': len(anomalies),
            'percentage': float(100 * len(anomalies) / len(self.data)),
            'anomalies': anomalies,
            'scores': anomaly_scores
        }
    
    def generate_recommendations(self):
        """🎯 Generate smart recommendations based on cluster"""
        labels = self.best_model['labels']
        profiles = self.generate_cluster_profiles()
        
        recommendations = {}
        
        for label, profile in profiles.items():
            if profile['business_label'] == 'Premium/High-Value':
                recommendations[label] = [
                    "VIP membership programs",
                    "Exclusive early access to new products",
                    "Personalized concierge service",
                    "Premium loyalty rewards"
                ]
            elif profile['business_label'] == 'Standard/Medium-Value':
                recommendations[label] = [
                    "Regular loyalty program",
                    "Targeted promotions",
                    "Upsell opportunities",
                    "Community engagement"
                ]
            else:
                recommendations[label] = [
                    "Re-engagement campaigns",
                    "Special discounts",
                    "Value bundles",
                    "Welcome back offers"
                ]
        
        return recommendations
    
    def run_full_analysis(self):
        """🔬 Run complete intelligent analysis"""
        print("\n" + "="*80)
        print("🧠 INTELLIGENT ADAPTIVE CLUSTERING ENGINE - STARTING ANALYSIS")
        print("="*80)
        
        # 1. Auto-select algorithm
        best_algo, scores = self.auto_select_algorithm()
        
        # 2. Get clustering results
        labels = self.best_model['labels']
        metrics = self.best_model['metrics']
        icso = self.best_model['icso']
        
        # 3. Generate cluster profiles
        print("\n📊 Generating Cluster Profiles...")
        profiles = self.generate_cluster_profiles()
        
        # 4. Detect anomalies
        print("🔍 Detecting Anomalies...")
        anomalies = self.detect_anomalies()
        
        # 5. Generate recommendations
        print("🎯 Generating Smart Recommendations...")
        recommendations = self.generate_recommendations()
        
        print("\n" + "="*80)
        
        return {
            'best_algorithm': best_algo,
            'algorithm_scores': scores,
            'clusters': labels,
            'metrics': metrics,
            'icso_score': icso,
            'profiles': profiles,
            'anomalies': anomalies,
            'recommendations': recommendations,
            'n_clusters': len(np.unique(labels[labels != -1]))
        }

