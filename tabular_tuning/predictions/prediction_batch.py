from base_constants import params_imbalanced_ensemble_adacost, ClassifierType, Classifier, \
    params_imbalanced_ensemble_adauboost, params_imbalanced_ensemble_balanced_cascade, \
    params_imbalanced_ensemble_balanced_random_forest, params_imbalanced_ensemble_easy_ensemble, \
    params_imbalanced_ensemble_kmeans_smote_boost, params_imbalanced_ensemble_overboost, \
    params_imbalanced_ensemble_self_paced, params_imbalanced_ensemble_smote_bagging, \
    params_imbalanced_ensemble_smote_boost
from base_functions import do_prediction, save_results
from tabular_tuning.tabular_tuning_functions import get_data

CLF = Classifier.IMBALANCED_ADACOST
CLF_TYPE = ClassifierType.SUPERVISED
CLF_PARAMS = params_imbalanced_ensemble_adacost

data, labels, groups = get_data()
result = do_prediction(data, labels,groups, CLF, CLF_TYPE, CLF_PARAMS, use_one_hot=True)
save_results(CLF.value, "predictions", result[0], result[1])

CLF = Classifier.IMBALANCED_ADAUBOOST
CLF_TYPE = ClassifierType.SUPERVISED
CLF_PARAMS = params_imbalanced_ensemble_adauboost

data, labels, groups = get_data()
result = do_prediction(data, labels,groups, CLF, CLF_TYPE, CLF_PARAMS, use_one_hot=True)
save_results(CLF.value, "predictions", result[0], result[1])

CLF = Classifier.IMBALANCED_BALANCED_CASCADE
CLF_TYPE = ClassifierType.SUPERVISED
CLF_PARAMS = params_imbalanced_ensemble_balanced_cascade

data, labels, groups = get_data()
result = do_prediction(data, labels,groups, CLF, CLF_TYPE, CLF_PARAMS, use_one_hot=True)
save_results(CLF.value, "predictions", result[0], result[1])


CLF = Classifier.IMBALANCED_BALANCED_RANDOM_FOREST
CLF_TYPE = ClassifierType.SUPERVISED
CLF_PARAMS = params_imbalanced_ensemble_balanced_random_forest

data, labels, groups = get_data()
result = do_prediction(data, labels,groups, CLF, CLF_TYPE, CLF_PARAMS, use_one_hot=True)
save_results(CLF.value, "predictions", result[0], result[1])


CLF = Classifier.IMBALANCED_EASY_ENSEMBLE
CLF_TYPE = ClassifierType.SUPERVISED
CLF_PARAMS = params_imbalanced_ensemble_easy_ensemble

data, labels, groups = get_data()
result = do_prediction(data, labels,groups, CLF, CLF_TYPE, CLF_PARAMS, use_one_hot=True)
save_results(CLF.value, "predictions", result[0], result[1])


CLF = Classifier.IMBALANCED_KMEANS_SMOTE_BOOST
CLF_TYPE = ClassifierType.SUPERVISED
CLF_PARAMS = params_imbalanced_ensemble_kmeans_smote_boost

data, labels, groups = get_data()
result = do_prediction(data, labels,groups, CLF, CLF_TYPE, CLF_PARAMS, use_one_hot=True)
save_results(CLF.value, "predictions", result[0], result[1])


CLF = Classifier.IMBALANCED_OVERBOOST
CLF_TYPE = ClassifierType.SUPERVISED
CLF_PARAMS = params_imbalanced_ensemble_overboost

data, labels, groups = get_data()
result = do_prediction(data, labels,groups, CLF, CLF_TYPE, CLF_PARAMS, use_one_hot=True)
save_results(CLF.value, "predictions", result[0], result[1])


'''
SELF PACED MA PROBLEM BEZAT V PIPELINE!!!!!!!
CLF = Classifier.IMBALANCED_SELF_PACED
CLF_TYPE = ClassifierType.SUPERVISED
CLF_PARAMS = params_imbalanced_ensemble_self_paced

data, labels, groups = get_data()
result = do_prediction(data, labels,groups, CLF, CLF_TYPE, CLF_PARAMS, use_one_hot=True)
save_results(CLF.value, "predictions", result[0], result[1])
'''

CLF = Classifier.IMBALANCED_SMOTE_BAGGING
CLF_TYPE = ClassifierType.SUPERVISED
CLF_PARAMS = params_imbalanced_ensemble_smote_bagging

data, labels, groups = get_data()
result = do_prediction(data, labels,groups, CLF, CLF_TYPE, CLF_PARAMS, use_one_hot=True)
save_results(CLF.value, "predictions", result[0], result[1])


CLF = Classifier.IMBALANCED_SMOTE_BOOST
CLF_TYPE = ClassifierType.SUPERVISED
CLF_PARAMS = params_imbalanced_ensemble_smote_boost

data, labels, groups = get_data()
result = do_prediction(data, labels, groups, CLF, CLF_TYPE, CLF_PARAMS, use_one_hot=True)
save_results(CLF.value, "predictions", result[0], result[1])