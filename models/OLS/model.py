import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class OLS:
    def __init__(self, n_epochs: int = 1000, learning_rate: float = 0.01):
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.w = 0

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        X = X.values.reshape(-1, 1)
        y = y.values.reshape(-1, 1)
        
        self.w = np.linalg.pinv(X.T @ X) @ X.T @ y

    def predict(self, X: pd.DataFrame):
        X = X.values.reshape(-1, 1)
        y_pred = X @ self.w

        return y_pred