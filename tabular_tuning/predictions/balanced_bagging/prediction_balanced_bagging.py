from base_constants import Classifier, ClassifierType, params_balanced_bagging
from base_functions import do_prediction, save_results
from tabular_tuning.tabular_tuning_functions import get_data


CLF = Classifier.BALANCED_BAGGING
CLF_TYPE = ClassifierType.SUPERVISED
CLF_PARAMS = params_balanced_bagging

data, labels, groups = get_data()
result = do_prediction(data, labels,groups, CLF, CLF_TYPE, CLF_PARAMS, use_one_hot=True)
save_results(CLF.value, "predictions", result[0], result[1])
