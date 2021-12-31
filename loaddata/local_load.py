import os
from constants import commonconstants


def get_data_file_path_list(type='classification',data_type='train'):
    full_file_path_list = []
    data_folder_path = commonconstants.CLASSIFICATION_DATA_FOLDER_PATH
    if type.lower() == 'regression':
        data_folder_path = commonconstants.REGRESSION_DATA_FOLDER_PATH
    data_folder_full_path = os.path.join(data_folder_path, data_type)
    filename_list = os.listdir(data_folder_full_path)
    for filename in filename_list:
        full_file_path = os.path.join(data_folder_full_path, filename)
        full_file_path_list.append(full_file_path)
    return full_file_path_list

