import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io


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
            data = pd.concat([X, y], axis=1)
            data = data.sample(frac=1).reset_index(drop=True)  
            X_shuffled = data.iloc[:, 0] 
            y_shuffled = data.iloc[:, 1]
            
            for pattern in range(0, len(X_shuffled)):
                self.predicts[pattern] = self.w0 + self.w1 * X_shuffled[pattern]
                self.errors[pattern] = y_shuffled[pattern] - self.predicts[pattern]

                self.w_history.append((self.w0, self.w1))
            
                self.w0 = self.w0 + self.learning_rate * self.errors[pattern]
                self.w1 = self.w1 + self.learning_rate * self.errors[pattern] * X_shuffled[pattern]

            self.mse_history.append(np.mean(np.square(self.errors))) 
            
    def predict(self, X: pd.DataFrame):
        y_pred = self.w0 + self.w1 * X
        return y_pred.to_numpy()
