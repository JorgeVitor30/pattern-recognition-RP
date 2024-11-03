import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class OLS:
    def __init__(self):
        self.w = 0
        
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        if (len(X) != len(y)): return False

        X = np.hstack((np.ones((X.shape[0], 1)), X.values.reshape(-1, 1)))
        y = y.values.reshape(-1, 1)
        
        self.w = np.linalg.pinv(X.T @ X) @ X.T @ y

    def predict(self, X: pd.DataFrame):
        X = np.hstack((np.ones((X.shape[0], 1)), X.values.reshape(-1, 1)))
        y_pred = X @ self.w

        return y_pred