import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from sklearn.utils import shuffle


class LogisticRegression:
    def __init__(self, n_epochs: int = 1000, learning_rate: float = 0.001):
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.w = None
        self.scaler_X = DataNormalizer()

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        if (len(X) != len(y)): return False
        
        self.w = np.zeros(X.shape[1])
        X = self.scaler_X.fit_transform(X)

        X = np.array(X)
        y = np.array(y)

        for _ in range(0, self.n_epochs):
            X, y = shuffle(X, y)
            for i in range(len(X)):
                xi = X[i]
                yi = y[i]

                error = yi - self._sigmoid(np.dot(xi, self.w))
                self.w += self.learning_rate * error * xi
            
    def predict(self, X: pd.DataFrame):
        X = self.scaler_X.transform(X)
        y_pred = self._sigmoid(np.dot(X, self.w))

        return np.round(y_pred)
    
    def calcule_metrics(self, y_real: np.ndarray, y_pred: np.ndarray):
        if (len(y_real) != len(y_pred)): return False

        if isinstance(y_real, pd.Series): y_real = y_real.to_numpy()
        if isinstance(y_pred, pd.Series): y_pred = y_pred.to_numpy()

        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        acurracy = 0

        for i in range(len(y_real)):
            if y_real[i] == 1 and y_pred[i] == 1:
                true_positive += 1
            elif y_real[i] == 0 and y_pred[i] == 1:
                false_positive += 1
            elif y_real[i] == 0 and y_pred[i] == 0:
                true_negative += 1
            elif y_real[i] == 1 and y_pred[i] == 0:
                false_negative += 1
            
            if (y_pred[i] == y_real[i]):
                acurracy +=1

        
        return {"TP": true_positive, "TN": true_negative, "FP": false_positive, "FN": false_negative, "AC": acurracy / len(y_pred)}
    

class DataNormalizer:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None

    def fit(self, data):
        self.min_ = np.min(data, axis=0)
        self.max_ = np.max(data, axis=0)

    def transform(self, data):
        if self.min_ is None or self.max_ is None:
            return
        data_normalized = (data - self.min_) / (self.max_ - self.min_)
        data_scaled = data_normalized * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        return data_scaled

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, normalized_data):
        if self.min_ is None or self.max_ is None:
            return
        normalized_data_original = (normalized_data - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        original_data = normalized_data_original * (self.max_ - self.min_) + self.min_
        return original_data