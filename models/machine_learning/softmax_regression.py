import numpy as np
from models.utils.activation_functions import softmax
from models.utils.loss_functions import categorical_crossentropy

class SoftmaxRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        """
        Initialize Softmax Regression model.

        Parameters:
        - learning_rate: The step size for gradient descent.
        - epochs: Number of iterations to train the model.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Train the Softmax Regression model using Gradient Descent.

        Parameters:
        - X: Input features (numpy array of shape [n_samples, n_features]).
        - y: Target labels (numpy array of shape [n_samples, n_classes]).
        """
        n_samples, n_features = X.shape
        n_classes = y.shape[1]
        
        # Initialize weights and bias
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)

        for epoch in range(self.epochs):
            # Compute linear model
            linear_model = np.dot(X, self.weights) + self.bias
            # Apply softmax activation
            y_pred = softmax(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y, axis=0)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Optional: Print loss every 100 epochs
            if epoch % 100 == 0:
                loss = categorical_crossentropy(y, y_pred)
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict_proba(self, X):
        """
        Predict class probabilities for input data.

        Parameters:
        - X: Input features (numpy array of shape [n_samples, n_features]).

        Returns:
        - Predicted probabilities (numpy array of shape [n_samples, n_classes]).
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return softmax(linear_model)

    def predict(self, X):
        """
        Predict class labels for input data.

        Parameters:
        - X: Input features (numpy array of shape [n_samples, n_features]).

        Returns:
        - Predicted class labels (numpy array of shape [n_samples]).
        """
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)
