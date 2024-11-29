import numpy as np
from models.utils.loss_functions import mean_squared_error

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        """
        Initialize Linear Regression model.

        Parameters:
        - learning_rate: The step size for gradient descent.
        - epochs: The number of iterations to train the model.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Train the model using Gradient Descent.

        Parameters:
        - X: Input features (numpy array of shape [n_samples, n_features]).
        - y: Target values (numpy array of shape [n_samples]).
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            # Calculate predictions
            y_pred = np.dot(X, self.weights) + self.bias

            # Calculate gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Optional: Print loss every 100 epochs
            if epoch % 100 == 0:
                loss = mean_squared_error(y, y_pred)
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        - X: Input features (numpy array of shape [n_samples, n_features]).

        Returns:
        - Predicted values (numpy array of shape [n_samples]).
        """
        return np.dot(X, self.weights) + self.bias
