import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from sklearn.utils import shuffle


class StochasticGradientDescent:
    def __init__(self, n_epochs: int = 1000, learning_rate: float = 0.001):
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.w0 = 0
        self.w1 = 0
        self.w_history = []
        self.mse_history = []

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        if (len(X) != len(y)): return False

        self.predicts = [0] * len(X)
        self.errors = [0] * len(X)

        for _ in range(0, self.n_epochs):
            X, y = shuffle(X, y)
            
            for pattern in range(0, len(X)):
                self.predicts[pattern] = self.w0 + self.w1 * X[pattern]
                self.errors[pattern] = y[pattern] - self.predicts[pattern]

                self.w_history.append((self.w0, self.w1))
            
                self.w0 = self.w0 + self.learning_rate * self.errors[pattern]
                self.w1 = self.w1 + self.learning_rate * self.errors[pattern] * X[pattern]

                self.mse_history.append(np.mean((y - self.predict(X)) ** 2)) 
            
    def predict(self, X: pd.DataFrame):
        y_pred = self.w0 + self.w1 * X
        return y_pred
