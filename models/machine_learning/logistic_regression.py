import numpy as np
from models.utils.activation_functions import sigmoid
from models.utils.loss_functions import binary_crossentropy

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        """
        Initialize Logistic Regression model.

        Parameters:
        - learning_rate: Step size for gradient descent.
        - epochs: Number of iterations to train the model.
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
            # Calculate linear model
            linear_model = np.dot(X, self.weights) + self.bias
            # Apply sigmoid activation
            y_pred = sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Optional: Print loss every 100 epochs
            if epoch % 100 == 0:
                loss = binary_crossentropy(y, y_pred)
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict_proba(self, X):
        """
        Predict probabilities for input data.

        Parameters:
        - X: Input features (numpy array of shape [n_samples, n_features]).

        Returns:
        - Probabilities (numpy array of shape [n_samples]).
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """
        Predict binary class labels for input data.

        Parameters:
        - X: Input features (numpy array of shape [n_samples, n_features]).
        - threshold: Threshold for classification.

        Returns:
        - Predicted class labels (numpy array of shape [n_samples]).
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
