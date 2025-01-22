import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt

class KMeansClustering:
    def __init__(self, min_clusters=4, max_clusters=20, n_init=20):
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.n_init = n_init
        self.scaler = StandardScaler()
        self.best_k = None
        self.best_model = None

    def fit(self, X):
        X_scaled = self.scaler.fit_transform(X)

        best_db_score = float('inf')
        best_model = None
        best_k = None

        for k in range(self.min_clusters, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=k, n_init=self.n_init, random_state=42)
            labels = kmeans.fit_predict(X_scaled)
            db_score = davies_bouldin_score(X_scaled, labels)

            if db_score < best_db_score:
                best_db_score = db_score
                best_model = kmeans
                best_k = k

        self.best_k = best_k
        self.best_model = best_model

        print(f"Número ótimo de clusters: {self.best_k} (DB score: {best_db_score:.4f})")

    def predict(self, X):
        if not self.best_model:
            raise ValueError("Model não foi treinado")

        X_scaled = self.scaler.transform(X)
        labels = self.best_model.predict(X_scaled)
        return labels

    def plot_clusters(self, X):
        if not self.best_model:
            raise ValueError("Model não foi treinado")

        X_scaled = self.scaler.transform(X)
        labels = self.best_model.predict(X_scaled)

        plt.figure(figsize=(10, 6))
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
        plt.scatter(
            self.best_model.cluster_centers_[:, 0],
            self.best_model.cluster_centers_[:, 1],
            c='red',
            marker='x',
            s=200,
            label='Centros'
        )
        plt.title(f"K-Means Com Cluster k={self.best_k}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.show()