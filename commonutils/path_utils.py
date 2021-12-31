import os
from constants import commonconstants


def get_downstream_baseline_model_path_from_datafile_path(data_file_path):
    data_filename_with_extension = data_file_path.rsplit(os.path.sep, 1)[1]
    data_filename_without_extension = data_filename_with_extension.replace(commonconstants.DEFAULT_TRAINING_DATA_FILE_EXTENSION,'')
    data_filename = data_filename_without_extension.rsplit('_', 1)[0]
    trained_model_name_list = os.listdir(commonconstants.TRAINED_MODEL_SAVE_FOLDER_PATH)
    for trained_model_name in trained_model_name_list:
        if trained_model_name.find(data_filename) != -1:
            return os.path.join(commonconstants.TRAINED_MODEL_SAVE_FOLDER_PATH, trained_model_name)
    return None


def get_correction_model_path_from_datafile_path(data_file_path,correction_model_type):
    data_filename_with_extension = data_file_path.rsplit(os.path.sep, 1)[1]
    data_filename_without_extension = data_filename_with_extension.replace(
        commonconstants.DEFAULT_TRAINING_DATA_FILE_EXTENSION, '')
    data_filename = data_filename_without_extension.rsplit('_', 1)[0]
    trained_model_name_list = os.listdir(commonconstants.TRAINED_MODEL_SAVE_FOLDER_PATH)
    for trained_model_name in trained_model_name_list:
        if trained_model_name.find(data_filename) != -1 and trained_model_name.find(correction_model_type) != -1:
            return os.path.join(commonconstants.TRAINED_MODEL_SAVE_FOLDER_PATH, trained_model_name)
    return None
