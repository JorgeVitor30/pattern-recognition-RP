import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class GaussianDiscriminant:
    def __init__(self):
        self.cov = {}
        self.means = {}
        self.priors = {} 
        self.scaler_X = DataNormalizer()
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        y = np.array(y).flatten()
        self.classes = np.unique(y) 
        linhas, colunas = X.shape
        X = self.scaler_X.fit_transform(X)

        for classe in self.classes:
            X_c = X[y == classe]
            self.means[classe] = np.mean(X_c, axis=0)
            self.priors[classe] = len(X_c) / linhas
            self.cov[classe] = np.atleast_2d(np.cov(X_c, rowvar=False))

    def predict(self, X: pd.DataFrame):
        X = self.scaler_X.transform(X)
        y_pred = []
        for _, row in X.iterrows():
            x = row.values 
            max_score = -float('inf')
            predicted_class = None

            for classe in self.classes:
                score = self._discriminant_function(x, self.means[classe], self.cov[classe], self.priors[classe])
                
                if score > max_score:
                    max_score = score
                    predicted_class = classe
            y_pred.append(predicted_class) 

        return np.array(y_pred) 

    def _discriminant_function(self, x, mean, cov, prior):
        diff = x - mean
        cov_inv = np.linalg.inv(cov)
        return -0.5 * np.dot(diff.T, np.dot(cov_inv, diff)) + np.log(prior)
    
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