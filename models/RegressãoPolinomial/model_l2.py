import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


class PolynomialRegressionL2:
    def __init__(self, number_polynomial: int = 1, lambda_: float = 0.01):
        self.np = number_polynomial 
        self.lambda_ = lambda_ 
        self.w = None 
        self.scaler_X = DataNormalizer()
        self.scaler_y = DataNormalizer()

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        if len(X) != len(y):
            raise ValueError("Número de registros diferente entre X e y")
                
        X_scaled = self.scaler_X.fit_transform(X) 
        X_transformed = self.tranform_columns(X_scaled)  
        X_transformed = np.hstack((np.ones((X_transformed.shape[0], 1)), X_transformed))  

        y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1))  
        
        I = np.eye(X_transformed.shape[1])  
        I[0, 0] = 0 
        self.w = np.linalg.pinv(X_transformed.T @ X_transformed + self.lambda_ * I) @ X_transformed.T @ y_scaled

    def predict(self, X: pd.DataFrame):              
        X_scaled = self.scaler_X.transform(X)
        X_transformed = self.tranform_columns(X_scaled)
        X_transformed = np.hstack((np.ones((X_transformed.shape[0], 1)), X_transformed)) 

        y_pred_scaled = X_transformed @ self.w
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled) 
        return y_pred

    def tranform_columns(self, X: np.ndarray):
        X_transformed = pd.DataFrame(X)
        
        for grau in range(2, self.np + 1):
            for coluna in X_transformed.columns:
                X_transformed[f"{coluna}^{grau}"] = X_transformed[coluna] ** grau
        
        return X_transformed.values

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