import numpy as np


def mean_absolute_error(predictions, actuals):
    return np.mean(np.abs(np.subtract(actuals, predictions)))


def mean_squared_error(predictions, actuals):
    return np.mean(np.square(np.subtract(actuals, predictions)))


def mean_absolute_percentage_error(predictions, actuals):
    return np.multiply(np.mean(np.abs(np.divide(np.subtract(actuals, predictions), actuals))), 100)


def symmetric_mean_absolute_percentage_error(predictions, actuals):
    return np.multiply(np.mean(np.divide(np.abs(np.subtract(actuals, predictions)), np.add(np.abs(actuals), np.abs(predictions)))), 100)
