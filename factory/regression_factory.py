from learn import deep_autoencoder, non_neural_regressors
from six import string_types


def get_regression_model(model_name='default',input_dimension=None,output_dimension=None):
    if model_name.lower() == 'default':
        return non_neural_regressors.get_random_forest_regressor()
    elif model_name.lower() == 'dae':
        if input_dimension is None or type(input_dimension) is string_types:
            raise AssertionError("The input dimension to autoencoders must be an integer")
        return deep_autoencoder.get_regression_autoencoder(input_dimension)
    return "Model not supported"
