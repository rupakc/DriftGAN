from loaddata import local_load
from pipeline import evaluate_downstream_model
from constants import commonconstants
from commonutils import path_utils


def evaluate_for_different_configurations(model_type):
    data_file_path_list = local_load.get_data_file_path_list(type=model_type, data_type='test')
    for data_file_path in data_file_path_list:
        model_path = path_utils.get_downstream_baseline_model_path_from_datafile_path(data_file_path)
        for correction_model in commonconstants.NOISE_CORRECTION_MODEL_LIST:
            for noise_type in commonconstants.NOISE_TYPE_LIST:
                for noise_level in commonconstants.NOISE_FACTOR_LIST:
                    evaluation_model = evaluate_downstream_model.EvaluateModelPipeline(data_path=data_file_path,
                                                                                       model_path=model_path, model_type=model_type,
                                                                                       correction_model=correction_model,
                                                                                       noise_level=noise_level,
                                                                                       noise_type=noise_type)
                    evaluation_model.execute_evaluation_pipeline()

