from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras import regularizers
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, LSTM, Bidirectional, RepeatVector, GRU


def get_regression_autoencoder(dimension, optimizer_name='adadelta', loss_function='kullback_leibler_divergence'):

    input_dimension = dimension
    input_data = Input(shape=(input_dimension,))

    encoded = Dense(int(input_dimension/2.0), activation='relu',activity_regularizer=regularizers.l1(10e-5))(input_data)
    encoded = Dropout(rate=0.3)(encoded)
    encoded = Dense(int(input_dimension/4.0), activation='relu')(encoded)

    decoded = Dense(int(input_dimension/2.0), activation='relu', activity_regularizer=regularizers.l1(10e-5))(encoded)
    decoded = Dropout(rate=0.3)(decoded)
    decoded = Dense(int(input_dimension), activation='sigmoid')(decoded)

    autoencoder = Model(input_data, decoded)
    autoencoder.compile(optimizer=optimizer_name, loss=loss_function, metrics=['mae'])
    return autoencoder


def get_classification_autoencoder(dimension, optimizer_name='adadelta', loss_function='mean_squared_error'):

    input_dimension = dimension
    input_data = Input(shape=(input_dimension,))

    encoded = Dense(int(input_dimension / 2.0), activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_data)
    encoded = Dropout(rate=0.3)(encoded)
    encoded = Dense(int(input_dimension / 4.0), activation='relu')(encoded)

    decoded = Dense(int(input_dimension / 2.0), activation='relu', activity_regularizer=regularizers.l1(10e-5))(encoded)
    decoded = Dropout(rate=0.3)(decoded)
    decoded = Dense(int(input_dimension), activation='sigmoid')(decoded)

    autoencoder = Model(input_data, decoded)
    autoencoder.compile(optimizer=optimizer_name, loss=loss_function, metrics=['mae'])
    return autoencoder


def get_deep_convolution_autoencoder(num_channels, img_height, img_width):

    input_img = Input(shape=(num_channels, img_height, img_width))  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='kullback_leibler_divergence')
    return autoencoder


def get_sequence_autoencoder_lstm(timesteps, input_dimension, latent_dimension):
    inputs = Input(shape=(timesteps, input_dimension))
    encoded = Bidirectional(LSTM(latent_dimension),merge_mode='ave')(inputs)
    decoded = RepeatVector(timesteps)(encoded)
    decoded = Bidirectional(LSTM(input_dimension, return_sequences=True),merge_mode='ave')(decoded)
    sequence_autoencoder = Model(inputs, decoded)
    return sequence_autoencoder


def get_sequence_autoencoder_gru(timesteps, input_dimension, latent_dimension):
    inputs = Input(shape=(timesteps, input_dimension))
    encoded = Bidirectional(GRU(latent_dimension),merge_mode='ave')(inputs)
    decoded = RepeatVector(timesteps)(encoded)
    decoded = Bidirectional(GRU(input_dimension, return_sequences=True),merge_mode='ave')(decoded)
    sequence_autoencoder = Model(inputs, decoded)
    return sequence_autoencoder
