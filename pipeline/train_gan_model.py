from keras.layers import Input, Dense, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from commonutils import matrix_utils, noiseutils
import numpy as np
import pandas as pd
import os
from constants import commonconstants


class TrainGAN:
    def __init__(self, data_filepath):
        self.optimizer = Adam(0.0002, 0.5)
        self.data_filepath = data_filepath
        self.input_dimension = self.get_input_dimension()

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'], optimizer=self.optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise_induced_data = Input(shape=(self.input_dimension,))
        noise_corrected_data = self.generator(noise_induced_data)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator(noise_corrected_data)

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model(noise_induced_data, valid)
        self.combined.compile(loss=['binary_crossentropy'], optimizer=self.optimizer)

    def get_train_data(self):
        train_dataframe = pd.read_csv(self.data_filepath, error_bad_lines=False)
        del train_dataframe['target']
        return train_dataframe.values

    def get_input_dimension(self):
        train_dataframe = pd.read_csv(self.data_filepath, error_bad_lines=False, nrows=10)
        num_columns = train_dataframe.values.shape[1]
        return num_columns-1

    def get_generator_discriminator_save_paths(self):
        data_filename_with_extension = self.data_filepath.rsplit(os.path.sep, 1)[1]
        data_filename_without_extension = data_filename_with_extension.replace(commonconstants.DEFAULT_TRAINING_DATA_FILE_EXTENSION, '')
        data_filename = data_filename_without_extension.rsplit('_', 1)[0]
        generator_filename = 'generator' + '_' + data_filename + '.h5'
        discriminator_filename = 'discriminator' + '_' + data_filename + '.h5'
        generator_filepath = os.path.join(commonconstants.TRAINED_MODEL_SAVE_FOLDER_PATH, generator_filename)
        discriminator_filepath = os.path.join(commonconstants.TRAINED_MODEL_SAVE_FOLDER_PATH, discriminator_filename)
        return generator_filepath, discriminator_filepath

    def build_generator(self):
        model = Sequential()
        model.add(Dense(512, input_dim=self.input_dimension))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(256))
        model.add(Dropout(rate=0.3))
        model.add(Dense(self.input_dimension, activation='sigmoid'))
        noise_induced_data = Input(shape=(self.input_dimension,))
        model_input = noise_induced_data
        noise_corrected_data = model(model_input)
        generator = Model(noise_induced_data, noise_corrected_data)
        generator.compile(optimizer=self.optimizer, loss='kullback_leibler_divergence', metrics=['mae'])
        generator.summary()
        return generator

    def build_discriminator(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.input_dimension))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        noise_corrected_data_or_real_data = Input(shape=(self.input_dimension,))
        model_input = noise_corrected_data_or_real_data
        output = model(model_input)
        return Model(noise_corrected_data_or_real_data, output)

    def train(self, epochs, batch_size=128):
        train_data = self.get_train_data()
        noise_induced_data = noiseutils.prepare_noise_injected_variations(train_data, noise_type='normal')
        original_train_data = matrix_utils.concat_matrix_n_times(train_data, len(commonconstants.NOISE_FACTOR_LIST))
        generator_filepath, discriminator_filepath = self.get_generator_discriminator_save_paths()

        # Adversarial ground truths
        valid_ground_truths = np.ones((batch_size, 1))
        fake_ground_truths = np.zeros((batch_size, 1))
        number_of_steps_per_epoch = original_train_data.shape[0]//batch_size
        print(number_of_steps_per_epoch)
        for epoch in range(epochs):
            start_index = 0
            stop_index = batch_size
            for step in range(number_of_steps_per_epoch):
                batch_original_data, batch_noise_induced_data = original_train_data[start_index:stop_index], \
                                                                noise_induced_data[start_index:stop_index]

                # Generate a half batch of new images
                gen_noise_corrected_data = self.generator.predict(batch_noise_induced_data)

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(batch_original_data, valid_ground_truths)
                d_loss_fake = self.discriminator.train_on_batch(gen_noise_corrected_data, fake_ground_truths)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Train the generator
                g_loss = self.combined.train_on_batch(batch_noise_induced_data, fake_ground_truths)
                start_index = start_index + batch_size
                stop_index = stop_index + batch_size

        self.generator.save(generator_filepath)
        self.discriminator.save(discriminator_filepath)


# train_filepath = 'C:\\Users\\rupachak\\Documents\\Github\\DriftGAN\\data\\regression\\test\\nyc_taxi_trip_duration_standard_test.csv'
# cgan = TrainGAN(train_filepath)
# cgan.train(epochs=501, batch_size=1024)
