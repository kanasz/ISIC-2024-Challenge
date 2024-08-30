from base_functions import do_prediction, save_results
from base_constants import params_imbalanced_ensemble_kmeans_smote_boost, ClassifierType, Classifier
from tabular_tuning.tabular_tuning_functions import get_data

CLF = Classifier.IMBALANCED_KMEANS_SMOTE_BOOST
CLF_TYPE = ClassifierType.SUPERVISED
CLF_PARAMS = params_imbalanced_ensemble_kmeans_smote_boost

data, labels, groups = get_data()
result = do_prediction(data, labels,groups, CLF, CLF_TYPE, CLF_PARAMS, use_one_hot=True)
save_results(CLF.value, "predictions", result[0], result[1])