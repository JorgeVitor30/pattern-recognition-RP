import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class GradientDescent:
    def __init__(self, n_epochs: int = 1000, learning_rate: float = 0.01):
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.w0 = 0
        self.w1 = 0
        self.w_history = []

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        if (len(X) != len(y)): return False

        self.predicts = [0] * len(X)
        self.errors = [0] * len(X)

        for _ in range(0, self.n_epochs):
            for pattern in range(0, len(X)):
                self.predicts[pattern] = self.w0 + self.w1 * X[pattern]
                self.errors[pattern] = y[pattern] - self.predicts[pattern]
            
            self.w_history.append((self.w0, self.w1))
            self.w0 = self.w0 + self.learning_rate * 1/len(X) * np.sum(self.errors)
            self.w1 = self.w1 + self.learning_rate * 1/len(X) * np.sum(self.errors * X)

    def predict(self, X: pd.DataFrame):
        y_pred = self.w0 + self.w1 * X
        return y_pred.to_numpy()

    def plot_regression_line(self, x, y, w0, w1, index):
        y_pred = self.predict = w0 + w1 * x  
        
        plt.figure()
        plt.scatter(x, y, color='blue')
        plt.plot(x, y_pred, color='red')  
        plt.title(f'Iteração {index}: w0={w0:.2f}, w1={w1:.2f}')
        plt.xlim(x.min() - 0.3, x.max() + 0.3)
        plt.ylim(y.min() - 0.3, y.max() + 0.3)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)

        plt.savefig(f"reports/figures/GradienteDescendente/pngs/frame_{index}.png")
        plt.close()

    def create_gif(self, frames, output_file, duration=100):
        imgs = [Image.open(frame) for frame in frames]
        imgs[0].save(output_file, save_all=True, append_images=imgs[1:], duration=duration, loop=0)


df = pd.read_csv("data/raw/artificial1d.csv", header=None)
X, y = df.iloc[:, 0], df.iloc[:, 1]

gd = GradientDescent()
gd.fit(X=X, y=y)

frames = []
for i, (w0, w1) in enumerate(gd.w_history):
    gd.plot_regression_line(X, y, w0, w1, i)
    frames.append(f"reports/figures/GradienteDescendente/pngs/frame_{i}.png")
gd.create_gif(frames, "reports/figures/GradienteDescendente/gif/regression_animation.gif", 50)
