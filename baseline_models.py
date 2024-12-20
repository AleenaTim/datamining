import numpy as np
import pandas as pd


# Baseline model functions
class BaselineModels:
    def __init__(self, data):
        # Initialize with the dataset
        self.data = data
        self.mean_vote = data['popularity'].mean()
        self.median_vote = data['popularity'].median()

        # vote_average-weighted average
        C = self.mean_vote
        M = data['vote_count'].quantile(0.9)
        self.weighted_scores = (
                                       data['vote_count'] * data['popularity'] + C * M
                               ) / (data['vote_count'] + M)

        self.simple_regression_model = None

        # Train simple regression model on vote_average
        if 'vote_average' in data.columns and not data['vote_average'].isna().all():
            from sklearn.linear_model import LinearRegression
            X = data[['vote_average']].fillna(0)
            y = data['popularity']
            self.simple_regression_model = LinearRegression()
            self.simple_regression_model.fit(X, y)

    def mean_model(self, row):
        """Predicts the mean popularity."""
        return self.mean_vote

    def median_model(self, row):
        """Predicts the median popularity."""
        return self.median_vote

    def vote_average_regression_model(self, row):
        """Predicts using a simple linear regression model on vote_average."""
        if self.simple_regression_model is not None and 'vote_average' in row:
            return self.simple_regression_model.predict([[row['vote_average']]])[0]
        return self.mean_vote


# Example Usage
if __name__ == "__main__":
    # Example dataset
    data = pd.DataFrame({
        'popularity': [20.1, 15.3, 35.2, 10.5, 18.7],
        'vote_count': [1000, 800, 1200, 400, 500],
        'vote_average': [7.5, 6.8, 8.3, 5.9, 7.0]
    })

    models = BaselineModels(data)

    # Example row
    row = {'vote_count': 600, 'vote_average': 25.0}

    print("Mean Model Prediction:", models.mean_model(row))
    print("Median Model Prediction:", models.median_model(row))
    # print("vote_average-Weighted Prediction:", models.vote_average_weighted_model(row))
    print("Simple Regression Prediction:", models.vote_average_regression_model(row))
