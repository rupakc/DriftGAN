from factory import classification_factory, regression_factory


def get_initialized_model_from_factory(model_operation_type,model_name,
                                       input_dimension=None, output_dimension=None):
    if model_operation_type.lower() == 'regression':
        return regression_factory.get_regression_model(model_name=model_name,
                                                       input_dimension=input_dimension,
                                                       output_dimension=output_dimension)
    elif model_operation_type.lower() == 'classification':
        return classification_factory.get_classification_model(model_name=model_name,
                                                               input_dimension=input_dimension,
                                                               output_dimension=output_dimension)
    return "Model not supported"
