from learn import deep_autoencoder, non_neural_classifiers
from six import string_types


def get_classification_model(model_name='default', input_dimension=None, output_dimension=None):
    if model_name.lower() == 'default':
        return non_neural_classifiers.get_random_forest_classifier()
    elif model_name.lower() == 'dae':
        if input_dimension is None or type(input_dimension) is string_types:
            raise AssertionError("The input dimension to autoencoders must be an integer")
        return deep_autoencoder.get_classification_autoencoder(input_dimension)
    return "Model not supported"
    # TODO - Add for GAN
