'''
1. Load test data
2. Load trained model
3. Predict for the test data
4. Evaluate the results
5. Persist the results in database
'''

from commonutils import logutils, model_persist_utils, dbutils, merge_utils
from constants import commonconstants
from evaluation import classification_metrics, regression_metrics, metric_helpers
import pandas as pd
from commonutils import noiseutils, path_utils, keras_model_load_utils

LOGGER = logutils.get_logger('Model Evaluation Pipeline')


class EvaluateModelPipeline:
    def __init__(self, data_path, model_path, model_type='classification',
                 correction_model='vanilla', noise_level=0.0, noise_type='none'):
        self.test_data_path = data_path
        self.trained_model_path = model_path
        self.model_type = model_type
        self.correction_model = correction_model
        self.noise_level = noise_level
        self.noise_type = noise_type

    def get_trained_model(self):
        return model_persist_utils.load_model(self.trained_model_path)

    def get_test_data(self):
        test_frame = pd.read_csv(self.test_data_path, error_bad_lines=False)
        return test_frame

    @staticmethod
    def get_model_predictions(trained_model,test_data):
        return trained_model.predict(test_data)

    def get_model_and_data_filename(self):
        full_model_name = self.trained_model_path.rsplit('\\', 1)[1]
        model_name = full_model_name.split('_', 1)[0]
        data_filename = full_model_name.split('_', 1)[1]
        data_filename = data_filename.replace('.pkl', '')
        return model_name, data_filename

    def get_model_corrected_data(self, noise_induced_data):
        correction_model_path = path_utils.get_correction_model_path_from_datafile_path(self.test_data_path,
                                                                                        self.correction_model)
        return keras_model_load_utils.get_model_predictions(correction_model_path, noise_induced_data)

    def get_metadata_dict(self):
        meta_data_dict = dict({})
        meta_data_dict['correction_model'] = self.correction_model
        meta_data_dict['noise_level'] = self.noise_level
        meta_data_dict['model_type'] = self.model_type
        meta_data_dict['noise_type'] = self.noise_type
        meta_data_dict['model_name'], meta_data_dict['data_filename'] = self.get_model_and_data_filename()
        return meta_data_dict

    def get_noise_injected_data(self,test_data):
        if self.noise_level == 0.0:
            return test_data
        return noiseutils.get_noise_injected_data(test_data, noise_factor=self.noise_level,
                                                noise_type=self.noise_type)

    def get_evaluation_results(self, trained_model, predicted_values, gold_values):
        summary_metric_dict = dict({})
        if self.model_type.lower() == commonconstants.CLASSIFICATION_MODEL_TYPE:
            accuracy_score = classification_metrics.get_accuracy_score(gold_values, predicted_values)
            precision_array, recall_array, f1_array, support_array = classification_metrics.get_precision_recall_f1_support(
                                                                        gold_values, predicted_values)
            class_iterable = trained_model.classes_
            precision_dict = metric_helpers.get_class_specific_metrics(class_iterable, precision_array)
            recall_dict = metric_helpers.get_class_specific_metrics(class_iterable, recall_array)
            f1_dict = metric_helpers.get_class_specific_metrics(class_iterable, f1_array)
            support_dict = metric_helpers.get_class_specific_metrics(class_iterable, support_array)
            summary_metric_dict['accuracy'] = accuracy_score
            summary_metric_dict['class_specific_support'] = support_dict
            summary_metric_dict['class_specific_precision'] = precision_dict
            summary_metric_dict['class_specific_recall'] = recall_dict
            summary_metric_dict['class_specific_f1'] = f1_dict
            summary_metric_dict['macro_precision'] = metric_helpers.get_macro_metrics(precision_array)
            summary_metric_dict['macro_recall'] = metric_helpers.get_macro_metrics(recall_array)
            summary_metric_dict['macro_f1'] = metric_helpers.get_macro_metrics(f1_array)
            summary_metric_dict['macro_support'] = metric_helpers.get_macro_metrics(support_array)
            return summary_metric_dict

        summary_metric_dict['r2_score'] = regression_metrics.get_r2_score(gold_values, predicted_values)
        summary_metric_dict['explained_variance'] = regression_metrics.get_explained_variance_score(gold_values, predicted_values)
        summary_metric_dict['mean_absolute_error'] = regression_metrics.get_mean_absolute_error(gold_values, predicted_values)
        summary_metric_dict['mean_squared_error'] = regression_metrics.get_mean_squared_error(gold_values, predicted_values)
        summary_metric_dict['mean_squared_log_error'] = regression_metrics.get_mean_squared_log_error(gold_values, predicted_values)
        summary_metric_dict['median_absolute_error'] = regression_metrics.get_median_absolute_error(gold_values, predicted_values)
        return summary_metric_dict

    @staticmethod
    def persist_results(dict_to_insert, db_connector):
        dbutils.insert_document(dict_to_insert, db_connector)

    def execute_evaluation_pipeline(self):
        trained_model = self.get_trained_model()
        LOGGER.info("Loaded Trained Model - %s" % self.trained_model_path)
        test_data_frame = self.get_test_data()
        LOGGER.info("Loaded Test data - %s" % self.test_data_path)
        test_values = test_data_frame['target'].values
        del test_data_frame['target']
        noise_induced_test_data = self.get_noise_injected_data(test_data_frame.values)
        LOGGER.info("Noise injected in the data")
        if self.correction_model.lower() != 'vanilla':
            noise_induced_test_data = self.get_model_corrected_data(noise_induced_test_data)
        LOGGER.info("Noise corrected data available")
        predicted_values = self.get_model_predictions(trained_model, noise_induced_test_data)
        summary_metric_dict = self.get_evaluation_results(trained_model, predicted_values, test_values)
        LOGGER.info("Evaluation Metrics Evaluated")
        meta_data_dict = self.get_metadata_dict()
        merged_dict = merge_utils.merge_dicts(meta_data_dict, summary_metric_dict)
        mongo_connector = dbutils.get_mongo_connector()
        self.persist_results(merged_dict, mongo_connector)
        print(merged_dict)
        LOGGER.info("Data persisted in database")


data_path = 'C:\\Users\\rupachak\\Documents\\Github\\DriftGAN\\data\\regression\\test\\nyc_taxi_trip_duration_standard_test.csv'
model_path = 'C:\\Users\\rupachak\\Documents\\Github\\DriftGAN\\models\\Random Forest_nyc_taxi_trip_duration_standard_train.pkl'


edm = EvaluateModelPipeline(data_path=data_path, model_path=model_path,
                            model_type='regression', noise_level=0.96, noise_type='normal',
                            correction_model='generator')

edm.execute_evaluation_pipeline()
