from keras.models import load_model


def get_model_predictions(trained_model_save_path, data):
    model = load_model(trained_model_save_path)
    return model.predict(data)

