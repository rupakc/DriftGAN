from pipeline import train_downstream_model
from loaddata import local_load
from commonutils import logutils
from constants import commonconstants

LOGGER = logutils.get_logger("Execute Train Pipeline")


def execute_model_train_pipeline(model_type='classification', data_type='train'):
    data_path_list = local_load.get_data_file_path_list(type=model_type, data_type=data_type)
    for data_path in data_path_list:
        train_model_object = train_downstream_model.TrainDownStreamModel(data_file_name=data_path,
                                                                         model_type=model_type)
        try:
            train_model_object.execute_model_train_pipeline()
            LOGGER.info("Executed train pipeline for file - %s and model type - %s " % (data_path, model_type))
        except Exception as e:
            LOGGER.error("Error occurred in execution pipeline %s : " % e.__str__())


if __name__ == '__main__':
    for model_type in commonconstants.MODEL_TYPE_LIST:
        try:
            execute_model_train_pipeline(model_type=model_type)
        except Exception as e:
            LOGGER.error("Error occurred in execution pipeline %s : " % e.__str__())
