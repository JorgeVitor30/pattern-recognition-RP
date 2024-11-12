import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

class PolynomialRegressionL2:
    def __init__(self, number_polynomial: int = 1, regularization_param: float = 0.01):
        self.np = number_polynomial
        self.lambda_ = regularization_param
        self.w = None 
        self.scaler = MinMaxScaler()

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        if len(X) != len(y):
            return
        
        original_columns = X.columns
        
        X = self.scaler.fit_transform(X)
        X = self.tranform_columns(X, original_columns)
        X = np.hstack((np.ones((X.shape[0], 1)), X)) 

        y = y.values.reshape(-1, 1)
        
        I = np.eye(X.shape[1])
        I[0, 0] = 0

        self.w = np.linalg.pinv(X.T @ X + self.lambda_ * I) @ X.T @ y

    def predict(self, X: pd.DataFrame):      
        original_columns = X.columns
        
        X = self.scaler.transform(X)
        X = self.tranform_columns(X, original_columns)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
 
        y_pred = X @ self.w
        return y_pred

    def tranform_columns(self, X: np.ndarray, original_columns: pd.Index):
        X_transformed = pd.DataFrame(X, columns=original_columns)
        
        for degree in range(2, self.np + 1):
            for column in X_transformed.columns:
                X_transformed[f"{column}^{degree}"] = X_transformed[column] ** degree
        
        return X_transformed.values
