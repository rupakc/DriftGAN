from sklearn.metrics.regression import mean_squared_error, median_absolute_error, mean_absolute_error
from sklearn.metrics.regression import r2_score, mean_squared_log_error, explained_variance_score


def get_mean_squared_error(gold_values, predicted_values):
    return mean_squared_error(gold_values, predicted_values)


def get_median_absolute_error(gold_values, predicted_values):
    return median_absolute_error(gold_values, predicted_values)


def get_mean_absolute_error(gold_values, predicted_values):
    return mean_absolute_error(gold_values, predicted_values)


def get_r2_score(gold_values, predicted_values):
    return r2_score(gold_values, predicted_values)


def get_mean_squared_log_error(gold_values, predicted_values):
    return mean_squared_log_error(gold_values, predicted_values)


def get_explained_variance_score(gold_values, predicted_values):
    return explained_variance_score(gold_values, predicted_values)
