from base_functions import do_prediction, save_results
from base_constants import ClassifierType, Classifier, params_imbalanced_ensemble_smote_bagging
from tabular_tuning.tabular_tuning_functions import get_data

# update me if necessary
CLF = Classifier.IMBALANCED_SMOTE_BAGGING
CLF_TYPE = ClassifierType.SUPERVISED
CLF_PARAMS = params_imbalanced_ensemble_smote_bagging

data, labels, groups = get_data()
result = do_prediction(data, labels,groups, CLF, CLF_TYPE, CLF_PARAMS, use_one_hot=True)
save_results(CLF.value, "predictions", result[0], result[1])