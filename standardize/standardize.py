import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Imputer
from sklearn.decomposition import PCA
import numpy as np
from six import string_types


def label_encode_frame(dataframe):
    columns = dataframe.columns
    encoder = LabelEncoder()
    for column in columns:
        if type(dataframe[column][0]) is str:
            dataframe[column] = encoder.fit_transform(dataframe[column].values)
    return dataframe


def get_feature_headers(num_features):
    feature_name_list = ['f_' + str(i+1) for i in range(num_features)]
    return feature_name_list


def get_standard_scaled_dataframe(dataframe):
    for column_name in dataframe.columns:
        standard_scaler = StandardScaler()
        dataframe[column_name] = standard_scaler.fit_transform(dataframe[column_name].values.reshape(-1, 1))
    return dataframe


def get_minmax_scaled_dataframe(dataframe):
    for column_name in dataframe.columns:
        standard_scaler = MinMaxScaler(feature_range=(0, 1))
        dataframe[column_name] = standard_scaler.fit_transform(dataframe[column_name].values.reshape(-1, 1))
    return dataframe


def get_imputed_dataframe(dataframe):
    for column_name in dataframe.columns:
        imputer = Imputer(strategy='median')
        dataframe[column_name] = imputer.fit_transform(dataframe[column_name].values.reshape(-1, 1))
    return dataframe


def drop_columns_from_df(dataframe, column_list_to_drop):
    dataframe.drop(column_list_to_drop, axis=1, inplace=True)
    return dataframe


def get_dimensionality_reduced_values_by_pca(dataframe, n_dimensions=100):
    pca = PCA(n_components=n_dimensions)
    reduced_dimensional_matrix = pca.fit_transform(dataframe.values)
    del dataframe
    return reduced_dimensional_matrix


def spilt_date(list_of_date_string, date_separator='-', time_separator=':'):
    month_list = list([])
    day_list = list([])
    year_list = list([])
    hour_list = list([])
    minute_list = list([])
    second_list = list([])
    for date_string in list_of_date_string:
        timestamp_list = date_string.strip().split(' ')
        date_list = timestamp_list[0].strip().split(date_separator)
        month_list.append(date_list[1])
        day_list.append(date_list[2])
        year_list.append(date_list[0])
        time_list = timestamp_list[1].strip().split(time_separator)
        hour_list.append(time_list[0])
        minute_list.append(time_list[1])
        second_list.append(time_list[2])
    return month_list, day_list, year_list, hour_list, minute_list, second_list
