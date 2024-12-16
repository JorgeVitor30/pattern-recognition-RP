import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class GaussianNaiveBayes:
    def __init__(self):
        self.vars = {}
        self.means = {}
        self.priors = {} 
        self.scaler_X = DataNormalizer()
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.classes = np.unique(y)
        linhas, colunas = X.shape

        for c in self.classes:
            X_c = X[y == c] 
            self.means[c] = np.mean(X_c, axis=0)
            self.vars[c] = np.var(X_c, axis=0)
            self.priors[c] = len(X_c) / linhas
    
    def predict(self, X: pd.DataFrame):
        y_pred = []
        for _, row in X.iterrows():
            x = row.values 
            class_probs = {}
            
            for c in self.classes:
                prior = np.log(self.priors[c])
                prob = 0
                
                for i in range(len(x)):
                    prob += np.log(self._calculate_verossimilhanca(x[i], self.means[c][i], self.vars[c][i]))
                
                class_probs[c] = prior + prob
            
            predicted_class = max(class_probs, key=class_probs.get)
            y_pred.append(predicted_class)
        
        return np.array(y_pred)


    def _calculate_verossimilhanca(self, x, mean, var):
        coeff = 1 / np.sqrt(2 * np.pi * var)
        exponent = np.exp(-(x - mean) ** 2 / (2 * var))
        return coeff * exponent
    
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