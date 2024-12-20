import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from baseline_models import BaselineModels

class SimpleV1Model:
    def __init__(self, data):
        """Initialize and train the model using selected features."""
        self.features = ['popularity', 'genre', 'vote_average', 'revenue']

        # Prepare features and target
        X = data[self.features]
        y = data['popularity']

        # Train the model
        self.model = LinearRegression()
        self.model.fit(X, y)

    def predict(self, row):
        """Make prediction for a single row."""
        X = np.array([row[feature] for feature in self.features]).reshape(1, -1)
        return self.model.predict(X)[0]

# Load data and train models
train_data = pd.read_csv("training_set.csv")
test_data = pd.read_csv("test_set.csv")

# Initialize and evaluate all models
baseline_models = BaselineModels(train_data)
v1_model = SimpleV1Model(train_data)

def calculate_mse(data, model_func):
    predictions = []
    for _, row in data.iterrows():
        predictions.append(model_func(row))
    return mean_squared_error(data['popularity'], predictions)

# Calculate MSE for all models
results = {
    'mean_model': {
        'train': calculate_mse(train_data, baseline_models.mean_model),
        'test': calculate_mse(test_data, baseline_models.mean_model)
    },
    'median_model': {
        'train': calculate_mse(train_data, baseline_models.median_model),
        'test': calculate_mse(test_data, baseline_models.median_model)
    },
    'vote_average_regression_model': {
        'train': calculate_mse(train_data, baseline_models.vote_average_regression_model),
        'test': calculate_mse(test_data, baseline_models.vote_average_regression_model)
    },
    'v1_model': {
        'train': calculate_mse(train_data, v1_model.predict),
        'test': calculate_mse(test_data, v1_model.predict)
    }
}

# Print results
print("\nMean Squared Error Results:")
print("-" * 50)
for model_name, scores in results.items():
    print(f"\n{model_name}:")
    print(f"Training MSE: {scores['train']:.4f}")
    print(f"Test MSE: {scores['test']:.4f}")

# Print feature coefficients for V1 model
print("\nV1 Model Coefficients:")
for feature, coef in zip(v1_model.features, v1_model.model.coef_):
    print(f"{feature}: {coef:.6f}")