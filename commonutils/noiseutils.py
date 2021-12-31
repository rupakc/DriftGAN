import numpy as np
from constants import commonconstants


def get_normal_noise_matrix(num_rows, dimension, mu=0.0, std=1.0):
    return np.random.normal(mu, std, (num_rows, dimension))


def get_uniform_noise_matrix(num_rows, dimension, lower_range_val=0.0, upper_range_val=1.0):
    return np.random.uniform(low=lower_range_val,
                             high=upper_range_val,
                             size=(num_rows, dimension))


def get_unique_random_integers(low, high, max_num):
    unique_set = set()
    while len(unique_set) < max_num:
        random_num_list = np.random.randint(low, high, max_num)
        for random_num in random_num_list:
            unique_set.add(random_num)
    return list(unique_set)


def get_noise_injected_data(data_matrix, noise_factor=commonconstants.DEFAULT_NOISE_FACTOR, noise_type='uniform',
                            normal_mean=0.0, normal_std=1.0, lower_range_val=0.0, upper_range_val=1.0):
    if noise_type == 'normal':
        return data_matrix + noise_factor * get_normal_noise_matrix(data_matrix.shape[0],
                                                                    data_matrix.shape[1],mu=normal_mean,
                                                                    std=normal_std)
    return data_matrix + noise_factor * get_uniform_noise_matrix(data_matrix.shape[0],
                                                                 data_matrix.shape[1],
                                                                 lower_range_val=lower_range_val,
                                                                 upper_range_val=upper_range_val)


def prepare_noise_injected_variations(data_matrix, noise_type='uniform', normal_mean=0.0, normal_std=1.0,
                                      lower_range_val=0.0, upper_range_val=1.0):
    noise_injected_matrix_list = list([])
    for noise_factor in commonconstants.NOISE_FACTOR_LIST:
        noise_injected_matrix = get_noise_injected_data(data_matrix, noise_factor, noise_type, normal_mean, normal_std, lower_range_val, upper_range_val)
        noise_injected_matrix_list.append(noise_injected_matrix)
    return np.vstack(tuple(noise_injected_matrix_list))
