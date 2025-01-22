import numpy as np
from sklearn.preprocessing import StandardScaler

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.components = None
        self.mean = None
        self.explained_variance_ratio = None

    def fit(self, X):
        X_scaled = self.scaler.fit_transform(X)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        U, S, Vt = np.linalg.svd(X_centered)

        self.components = Vt[:self.n_components]

        total_variance = np.sum(S**2)
        explained_variance = S[:self.n_components]**2
        self.explained_variance_ratio = explained_variance / total_variance

    def predict(self, X):
        if self.components is None:
            raise ValueError("O modelo PCA não foi ajustado. Use o método fit primeiro.")
        
        X_scaled = self.scaler.fit_transform(X)
        X_centered = X - self.mean
        
        return np.dot(X_centered, self.components.T)

    def explained_variance(self):
        if self.explained_variance_ratio is None:
            raise ValueError("O modelo PCA não foi ajustado. Use o método fit primeiro.")
        return self.explained_variance_ratio
