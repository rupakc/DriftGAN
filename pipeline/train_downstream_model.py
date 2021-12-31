'''
1. Fetch training data
2. Get initialized model for training
3. Train the model by calling fit
4. Persist the trained model
'''

import pandas as pd
from learn import non_neural_classifiers, non_neural_regressors
from commonutils import model_persist_utils
import os
from constants import commonconstants
from commonutils import logutils

LOGGER = logutils.get_logger("TrainDownstreamModel")


class TrainDownStreamModel:
    def __init__(self,data_file_name,model_type):
        self.data_file_name = data_file_name
        self.model_type = model_type

    def get_train_dataframe(self):
        dataframe = pd.read_csv(self.data_file_name, error_bad_lines=False)
        return dataframe

    def get_initialized_model(self):
        if self.model_type.lower() == 'classification':
            return non_neural_classifiers.get_random_forest_classifier()
        elif self.model_type.lower() == 'regression':
            return non_neural_regressors.get_random_forest_regressor()

    @staticmethod
    def train_model(model, feature_set, target_set):
        model.fit(feature_set, target_set)

    @staticmethod
    def persist_trained_model(trained_model, filename_to_save):
        model_persist_utils.save_model(trained_model, filename_to_save)

    @staticmethod
    def get_filename_to_save_model(model_name, data_filepath):
        data_filename_with_extension = str(data_filepath.rsplit(os.sep, 1)[1])
        data_filename = str(data_filename_with_extension.rsplit('.',1)[0])
        filename_to_save = commonconstants.TRAINED_MODEL_SAVE_FOLDER_PATH + model_name + '_' + \
                           data_filename + commonconstants.TRAINED_MODEL_BINARY_EXTENSION
        return filename_to_save

    def execute_model_train_pipeline(self):
        dataframe = self.get_train_dataframe()
        LOGGER.info("Dataframe fetched for data file - %s and model type - %s" %(self.data_file_name, self.model_type))
        model_list, model_name_list = self.get_initialized_model()
        LOGGER.info("Model initialized for data file - %s and model type - %s" % (self.data_file_name, self.model_type))
        target_values = dataframe['target'].values
        del dataframe['target']
        for model, model_name in zip(model_list, model_name_list):
            self.train_model(model, dataframe.values, target_values)
            LOGGER.info("Model trained for data file - %s and model type - %s" % (self.data_file_name, self.model_type))
            filename_to_save = self.get_filename_to_save_model(model_name, self.data_file_name)
            self.persist_trained_model(model, filename_to_save)
            LOGGER.info("Model persisted for data file - %s and model type - %s" % (self.data_file_name, self.model_type))
