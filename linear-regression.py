import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearRegressionModel:
    def __init__(self, learning_rate=0.00001, epochs=1500, history_interval=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.history_interval = history_interval
        self.weight = 0
        self.bias = 0

    def _gradient_descent(self, w_now, b_now, points, x_col, y_col):
        if x_col is None and y_col is None:
            columns = points.columns.to_list()
            x_col, y_col = columns[0], columns[1]

        w_gradient = 0
        b_gradient = 0
        n = len(points)

        for i in range(n):
            x = points.iloc[i][x_col]
            y = points.iloc[i][y_col]

            residual = y - (w_now * x + b_now)
            w_gradient += (-2 / n) * x * residual
            b_gradient += (-2 / n) * residual

        w = w_now - self.learning_rate * w_gradient
        b = b_now - self.learning_rate * b_gradient
        return w, b

    def fit(self, points, x_col=None, y_col=None):
        if x_col is None and y_col is None:
            columns = points.columns.to_list()
            x_col, y_col = columns[0], columns[1]

        self.weight, self.bias = 0, 0
        self.history = {}
        for i in range(self.epochs):

            if i % self.history_interval == 0:
                error = self.mse(points, self.weight, self.bias)
                self.history[i] = error
                print(f"I'm here {i}: {error}")

            self.weight, self.bias = self._gradient_descent(
                self.weight, self.bias, points, x_col, y_col)
        return self

    def get_history(self):
        plt.figure(figsize=(4, 3))
        plt.plot(self.history.keys(), self.history.values())
        plt.title('Cost History')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.show()

    def mse(self, points, w, b, x_col=None, y_col=None):
        if x_col is None and y_col is None:
            columns = points.columns.to_list()
            x_col, y_col = columns[0], columns[1]

        total_error = 0
        for i in range(len(points)):
            x = points.iloc[i][x_col]
            y = points.iloc[i][y_col]
            total_error += (y - (w*x + b)) ** 2
        return total_error / len(points)


model = LinearRegressionModel()

points = pd.read_csv("IceCreamData.csv")
model.fit(points)
model.get_history()
