import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


class KNN:
    def __init__(self, k: int = 2):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X: pd.DataFrame):
        X = np.array(X)
        predictions = []
        
        for x_test in X:
            distances = []
            for x_train in self.X_train:
                distance = np.sqrt(np.sum((x_test - x_train) ** 2))
                distances.append(distance)
            
            k_indices = np.argsort(distances)[:self.k]
            
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common)
        
        return np.array(predictions)
    
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