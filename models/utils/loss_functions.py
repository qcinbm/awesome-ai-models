import numpy as np

def mean_squared_error(y_true, y_pred):
    """
    Computes Mean Squared Error (MSE) between true and predicted values.
    """
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    """
    Computes Mean Absolute Error (MAE) between true and predicted values.
    """
    return np.mean(np.abs(y_true - y_pred))

def binary_crossentropy(y_true, y_pred):
    """
    Computes Binary Cross-Entropy loss.
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_crossentropy(y_true, y_pred):
    """
    Computes Categorical Cross-Entropy loss.
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Computes Huber loss, combining MSE and MAE for robust error handling.
    """
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * error ** 2
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.mean(np.where(is_small_error, squared_loss, linear_loss))

def cross_entropy_loss(y_true, y_pred):
    """
    Computes Cross Entropy Loss for binary or categorical classification.
    Parameters and Returns are same as explained above.
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    if len(y_true.shape) == 2:
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    else:
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss
