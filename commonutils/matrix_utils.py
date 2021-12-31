import numpy as np
import copy


def concat_matrix_n_times(data_matrix,num_times):
    copy_matrix = copy.deepcopy(data_matrix)
    for i in range(num_times-1):
        copy_matrix = np.append(copy_matrix,data_matrix,axis=0)
    return copy_matrix
