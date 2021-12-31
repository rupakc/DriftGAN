import pandas as pd
import os
from constants import commonconstants
from sklearn.model_selection import train_test_split


def generate_train_test_files(type='classification'):
    data_folder_path = commonconstants.CLASSIFICATION_DATA_FOLDER_PATH
    if type.lower() == 'regression':
        data_folder_path = commonconstants.REGRESSION_DATA_FOLDER_PATH
    filename_list = os.listdir(data_folder_path)
    for filename in filename_list:
        file_name_without_extension = str(filename.rsplit('.', 1)[0])
        full_file_path = os.path.join(data_folder_path, filename)
        dataframe = pd.read_csv(full_file_path)
        train_df, test_df = train_test_split(dataframe, test_size=0.2, random_state=42)
        train_file_name = file_name_without_extension + '_' + 'train' + '.csv'
        test_file_name = file_name_without_extension + '_' + 'test' + '.csv'
        full_train_file_path = os.path.join(data_folder_path,train_file_name)
        full_test_file_path = os.path.join(data_folder_path, test_file_name)
        train_df.to_csv(full_train_file_path, index=False, encoding='utf-8')
        test_df.to_csv(full_test_file_path, index=False, encoding='utf-8')
