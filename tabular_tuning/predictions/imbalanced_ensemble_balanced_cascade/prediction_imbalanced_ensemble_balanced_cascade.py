from tabular_tuning.tabular_tuning_functions import get_data
from base_functions import do_prediction, save_results, ClassifierType, Classifier
from base_constants import params_imbalanced_ensemble_balanced_cascade

CLF = Classifier.IMBALANCED_BALANCED_CASCADE
CLF_TYPE = ClassifierType.SUPERVISED
CLF_PARAMS = params_imbalanced_ensemble_balanced_cascade

data, labels, groups = get_data()
result = do_prediction(data, labels,groups, CLF, CLF_TYPE, CLF_PARAMS, use_one_hot=True)
save_results(CLF.value, "predictions", result[0], result[1])
