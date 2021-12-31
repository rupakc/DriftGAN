from commonutils import noiseutils
import numpy as np
from commonutils import matrix_utils
from constants import commonconstants
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard


def noise_injected_data_generator(X_data,batch_size):
    samples_per_epoch = X_data.shape[0]
    number_of_batches = samples_per_epoch//batch_size
    counter = 0
    while True:
        X_batch = np.array(X_data[batch_size*counter:batch_size*(counter+1)])
        X_batch_noise_input = noiseutils.prepare_noise_injected_variations(X_batch)
        X_batch_output = matrix_utils.concat_matrix_n_times(X_batch,len(commonconstants.NOISE_FACTOR_LIST))
        counter = counter + 1
        yield X_batch_noise_input, X_batch_output
        if counter >= number_of_batches:
            counter = 0


def get_callback_list(model_filepath_save, log_directory_path, batch_size):
    early_stop = EarlyStopping(patience=7)
    model_checkpoint = ModelCheckpoint(filepath=model_filepath_save,save_best_only=True)
    reduce_lr_on_plateau = ReduceLROnPlateau()
    tensorboard_logs = TensorBoard(log_dir=log_directory_path,write_graph=False,
                                   write_images=True,batch_size=batch_size)
    return [early_stop, model_checkpoint, reduce_lr_on_plateau, tensorboard_logs]

