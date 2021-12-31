'''
1. Load Train and Test data
2. Get initialized Neural Model
3. Fit the model with generator and callbacks
4. Save the history of the model in db
'''


from commonutils import keras_fit_utils, dbutils, logutils
from constants import commonconstants, dbconstants
import pandas as pd
from factory import model_factory
import os

LOGGER = logutils.get_logger("Generative Neural Model")


class TrainNeuralGenerativeModel:
    def __init__(self, train_data_path, test_data_path, train_batch_size,
                 test_batch_size, model_operation_type='regression', nn_model_type='dae'):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.model_operation_type = model_operation_type
        self.nn_model_type = nn_model_type
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

    def get_train_and_validation_data(self):
        train_frame = pd.read_csv(self.train_data_path, error_bad_lines=False)
        test_frame = pd.read_csv(self.test_data_path, error_bad_lines=False)
        del train_frame['target']
        del test_frame['target']
        return train_frame.values, test_frame.values

    def get_metadata_dict(self):
        meta_data_dict = dict({})
        meta_data_dict['model_operation_type'] = self.model_operation_type
        meta_data_dict['nn_model_type'] = self.nn_model_type
        meta_data_dict['train_batch_size'] = self.train_batch_size
        meta_data_dict['test_batch_size'] = self.test_batch_size
        data_file_name_with_extension = self.train_data_path.rsplit(os.path.sep, 1)[1]
        data_file_name_without_extension = data_file_name_with_extension.replace('.csv', '')
        data_file_name = data_file_name_without_extension.rsplit('_', 1)[0]
        meta_data_dict['data_filename'] = data_file_name
        return meta_data_dict

    def get_model_save_and_log_path(self):
        data_file_name_with_extension = self.train_data_path.rsplit(os.path.sep,1)[1]
        data_file_name_without_extension = data_file_name_with_extension.replace('.csv', '')
        data_file_name = data_file_name_without_extension.rsplit('_', 1)[0]
        model_save_file_name = self.nn_model_type + '_' + data_file_name + '.h5'
        model_save_path = os.path.join(commonconstants.TRAINED_MODEL_SAVE_FOLDER_PATH, model_save_file_name)
        model_log_path = commonconstants.LOG_DIR_PATH
        return model_save_path, model_log_path

    def get_initialized_neural_model(self,input_dimension,output_dimension=None):
        return model_factory.get_initialized_model_from_factory(self.model_operation_type, self.nn_model_type,
                                                         input_dimension=input_dimension,
                                                         output_dimension=output_dimension)

    def fit_initialized_neural_model(self, model, train_data, validation_data, model_save_path, model_log_path):

        callback_list = keras_fit_utils.get_callback_list(model_save_path, model_log_path, self.train_batch_size)
        h = model.fit_generator(
                            generator=keras_fit_utils.noise_injected_data_generator(train_data, self.train_batch_size),
                            epochs=commonconstants.DEFAULT_NUMBER_OF_EPOCHS,
                            steps_per_epoch=train_data.shape[0]//self.train_batch_size,
                            validation_data=keras_fit_utils.noise_injected_data_generator(validation_data, self.test_batch_size),
                            validation_steps=validation_data.shape[0]//self.test_batch_size,
                            callbacks=callback_list
                            )
        return h.history

    @staticmethod
    def persist_training_results_in_db(db_connector, result_dictionary):
        dbutils.insert_document_training(result_dictionary, db_connector)

    def execute_model_train_pipeline(self):
        train_data, test_data = self.get_train_and_validation_data()
        LOGGER.info("Train and Test data loaded in memory")
        model = self.get_initialized_neural_model(input_dimension=train_data.shape[1])
        model_save_path, model_log_path = self.get_model_save_and_log_path()
        LOGGER.info("Neural Net Initialized")
        history_dict = self.fit_initialized_neural_model(model=model, train_data=train_data,
                                          validation_data=test_data, model_save_path=model_save_path,
                                          model_log_path=model_log_path)
        LOGGER.info("Neural Network Trained and saved")
        meta_data_dict = self.get_metadata_dict()
        meta_data_dict['history'] = str(history_dict)
        meta_data_dict['model_config'] = str(model.to_json())
        db_connector = dbutils.get_mongo_connector(collection_name=dbconstants.MODEL_STORE_COLLECTION)
        self.persist_training_results_in_db(db_connector, meta_data_dict)
        LOGGER.info("Neural Training Data Persisted in database")


train_path = 'C:\\Users\\rupachak\\Documents\\Github\\DriftGAN\\data\\classification\\train\\walmart_trip_type_challenge_standard_train.csv'
test_path = 'C:\\Users\\rupachak\\Documents\\Github\\DriftGAN\\data\\classification\\test\\walmart_trip_type_challenge_standard_test.csv'

tng = TrainNeuralGenerativeModel(train_data_path=train_path, test_data_path=test_path,
                                 train_batch_size=1024, test_batch_size=512,
                                 model_operation_type='classification')

tng.execute_model_train_pipeline()
