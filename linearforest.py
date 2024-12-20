import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

class MiniLinearForest:
    def __init__(self, nearest_segments=2, nearest_weight=0.4, rest_weight=0.6):
        self.intercepts =[]
        self.slopes =[]
        self.nearest_weight = nearest_weight
        self.rest_weight = rest_weight
        self.nearest_segments = nearest_segments

    def train(self, data):
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        x = data['X'].values.reshape(-1, 1)
        y = data['Y'].values.reshape(-1, 1)
        sorted_indices = np.argsort(x.flatten())
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]
        for i in range(len(x_sorted) - 1):
            lr = LinearRegression()
            lr.fit(x_sorted[i:i+2], y_sorted[i:i+2])
            self.intercepts.append(lr.intercept_[0])
            self.slopes.append(lr.coef_[0][0])

    def predict(self, new_x):
        predicted_prices = []
        for area in new_x:
            nearest_weighted_sum = 0
            rest_weighted_sum = 0
            nearest_count = 0
            rest_count = 0
            for i, slope in enumerate(self.slopes):
                predicted_price = self.intercepts[i] + slope * area
                if i < self.nearest_segments: 
                    nearest_weighted_sum += self.nearest_weight * predicted_price
                    nearest_count += 1
                else:
                    rest_weighted_sum += self.rest_weight * predicted_price
                    rest_count += 1
            if nearest_count > 0 and rest_count > 0:
                weighted_avg_prediction = (nearest_weighted_sum / nearest_count) + (rest_weighted_sum / rest_count)
            elif nearest_count > 0:
                weighted_avg_prediction = nearest_weighted_sum / nearest_count
            else:
                weighted_avg_prediction = rest_weighted_sum / rest_count
            predicted_prices.append(weighted_avg_prediction)            
        return predicted_prices
    def plot_predictions(self, data, new_x, predicted_prices):
        plt.figure(figsize=(8, 6))
        plt.scatter(data['X'], data['Y'], label='Original Data')
        plt.title('Piecewise Linear Model with Predicted Values (Weighted Aggregation)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        for i in range(len(self.intercepts)):
            plt.plot([data['X'].iloc[i], data['X'].iloc[i+1]], 
                     [self.intercepts[i] + self.slopes[i] * data['X'].iloc[i], self.intercepts[i] + self.slopes[i] * data['X'].iloc[i+1]], 
                     c='r')

        plt.scatter(new_x, predicted_prices, c='g', marker='o', label='Predicted Values')
        plt.legend()
        plt.show()