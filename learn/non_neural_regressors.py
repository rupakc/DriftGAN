from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor, ElasticNet
from sklearn.svm import LinearSVR


def get_ensemble_tree_regressors():
    rf = RandomForestRegressor(n_estimators=51,random_state=42)
    grad = GradientBoostingRegressor(random_state=42)
    ada = AdaBoostRegressor(random_state=42)
    extra = ExtraTreesRegressor(n_estimators=51,random_state=42)
    classifier_list = [rf, grad, ada, extra]
    classifier_name_list = ["Random Forest","GradientBoost", "AdaBoost", "ExtraTrees"]
    return classifier_list, classifier_name_list


def get_discriminative_regressor():
    support_vector = LinearSVR(random_state=42)
    return [support_vector], ["Support Vector Machine (Linear Kernel)"]


def get_linear_type_regressor():
    linear_reg = LinearRegression()
    elastic_net = ElasticNet()
    return [linear_reg, elastic_net], ["Logistic Regression", "Elastic Net"]


def get_sgd_regressor():
    sgd = SGDRegressor(random_state=42)
    return [sgd], ["Stochastic Gradient Descent"]


def get_random_forest_regressor():
    rf = RandomForestRegressor(n_estimators=51, random_state=42)
    return [rf], ['Random Forest']


def get_ada_boost_regressor():
    ada = AdaBoostRegressor(random_state=42)
    return [ada], ['AdaBoost']


def get_grad_boost_regressor():
    grad = GradientBoostingRegressor(random_state=42, warm_start=True)
    return [grad], ['GradientBoost']


def get_extra_trees_regressor():
    extra = ExtraTreesRegressor(n_estimators=51, random_state=42)
    return [extra], ['ExtraTrees']


def get_elasticnet_regressor():
    elastic_net = ElasticNet()
    return [elastic_net], ['ElasticNet']


def get_linear_regressor():
    linear_reg = LinearRegression()
    return [linear_reg], ['LinearReg']


def get_all_classifiers():
    regressor_list, regressor_name_list = get_ensemble_tree_regressors()
    temp_regressor_list, temp_regressor_name_list = get_discriminative_regressor()
    regressor_list.extend(temp_regressor_list)
    regressor_name_list.extend(temp_regressor_name_list)
    temp_regressor_list, temp_regressor_name_list = get_linear_type_regressor()
    regressor_list.extend(temp_regressor_list)
    regressor_name_list.extend(temp_regressor_name_list)
    return regressor_list, regressor_name_list
