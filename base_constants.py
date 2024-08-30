from enum import Enum


# enum type to differentiate classifier type
class ClassifierType(Enum):
    SUPERVISED = 'supervised'
    SEMI_SUPERVISED = 'semi_supervised'


class SearchType(Enum):
    GRID_SEARCH = 'grid_search'
    RANDOM_SEARCH = 'random_search'


class MetricType(Enum):
    ACCURACY = "accuracy"
    G_MEAN = "geometric_mean"
    ROC_AUC = "roc_auc"
    SENSITIVITY = "sensitivity"
    SPECIFICITY = "specificity"
    F1 = "f1"
    pAUC = "pAUC"


# enum type to differentiate classifiers
class Classifier(Enum):
    ALL = "all"
    ADABOOST = "adaboost"
    BALANCED_BAGGING = "balanced_bagging"
    BALANCED_RANDOM_FOREST = "balanced_random_forest"
    EASY_ENSEMBLE = "easy_ensemble"
    ISOLATION_FOREST = "isolation_forest"
    LEAST_SQUARES_ANOMALY_DETECTION = "least_squares_anomaly_detection"
    ONE_CLASS_SVM = "ocsvm"
    RANDOM_FOREST = "random_forest"
    RUS_BOOST = "rusboost"
    PYOD_CBLOF = "pyod_cblof"
    PYOD_COPOD = "pyod_copod"
    PYOD_COF = "pyod_cof"
    PYOD_KNN = "pyod_knn"
    PYOD_LMDD = "pyod_lmdd"
    PYOD_LOCI = "pyod_loci"
    PYOD_LOF = "pyod_lof"
    PYOD_MCD = "pyod_mcd"
    PYOD_SOS = "pyod_sos"
    PYOD_SOD = "pyod_sod"
    PYOD_XGBOD = "pyod_xgbod"
    PYOD_HBOS = "pyod_hbos"
    PYOD_LODA = "pyod_loda"
    PYOD_IF = "pyod_isolation_forest"
    PYOD_FEATURE_BAGGING = "pyod_feature_bagging"
    PYOD_PCA = "pyod_pca"
    SAMPLING_SMOTE_SVC = "sampling_smote_svc"
    SAMPLING_SMOTE_XGBOOST = "sampling_smote_xgboost"
    SAMPLING_ADASYN_SVC = "sampling_adasyn_svc"
    SAMPLING_ADASYN_XGBOOST = "sampling_adasyn_xgboost"
    SAMPLING_BORDERLINESMOTE_SVC = "sampling_borderlinesmote_svc"
    SAMPLING_BORDERLINESMOTE_XGBOOST = "sampling_borderlinesmote_xgboost"
    SAMPLING_SMOTEN_SVC = "sampling_smoten_svc"
    SAMPLING_SMOTEN_XGBOOST = "sampling_smoten_xgboost"
    SAMPLING_NEAR_MISS_SVC = "sampling_near_miss"
    SAMPLING_NEAR_MISS_XGBOOST = "sampling_near_miss_xgboost"
    SAMPLING_TOMEK_LINKS_SVC = "sampling_tomek_links"
    SAMPLING_TOMEK_LINKS_XGBOOST = "sampling_tomek_links_xgboost"
    SAMPLING_CLUSTER_CENTROIDS_SVC = "sampling_cluster_centroids"
    SAMPLING_CLUSTER_CENTROIDS_XGBOOST = "sampling_cluster_centroids_xgboost"
    SAMPLING_COMBINE_SMOTE_TOMEK_SVC = "sampling_combine_smote_tomek_svc"
    SAMPLING_COMBINE_SMOTE_TOMEK_XGBOOST = "sampling_combine_smote_tomek_xgboost"
    SAMPLING_COMBINE_SMOTE_ENN_SVC = "sampling_combine_smote_enn_svc"
    SAMPLING_COMBINE_SMOTE_ENN_XGBOOST = "sampling_combine_smote_enn_xgboost"
    IMBALANCED_BALANCED_CASCADE = "imbalanced_ensemble_balanced_cascade"
    IMBALANCED_SELF_PACED = "imbalanced_ensemble_self_paced"
    IMBALANCED_BALANCED_RANDOM_FOREST = "imbalanced_ensemble_balanced_random_forest"
    IMBALANCED_EASY_ENSEMBLE = "imbalanced_ensemble_easy_ensemble"
    IMBALANCED_UNDERBAGGING = "imbalanced_ensemble_underbagging"
    IMBALANCED_OVERBOOST = "imbalanced_ensemble_overboost"
    IMBALANCED_SMOTE_BOOST = "imbalanced_ensemble_smote_boost"
    WEIGHTED_SVC = "weighted_svc"
    WEIGHTED_RANDOM_FOREST = "weighted_random_forest"
    IMBALANCED_KMEANS_SMOTE_BOOST = "imbalanced_ensemble_kmeans_smote_boost"
    IMBALANCED_SMOTE_BAGGING = "imbalanced_ensemble_smote_bagging"
    IMBALANCED_ADAUBOOST = "imbalanced_ensemble_adauboost"
    IMBALANCED_ADACOST = "imbalanced_ensemble_adacost"


# random_state param constant to be able to reproduce achieved results
RANDOM_STATE = 1000
N_ITERATIONS = 1
# determined by us
DEFAULT_CONTAMINATION = 0.2

########################################################################################################################
# param dictionaries

params_pyod_pca = {
    "model__contamination": [DEFAULT_CONTAMINATION],
    "model__svd_solver": ['auto', 'full', 'arpack', 'randomized']
}

params_pyod_feature_bagging = {
    "model__contamination": [DEFAULT_CONTAMINATION],
    "model__n_estimators": [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    "model__max_features": [0.5, 0.7, 1.0],
    "model__combination": ['average', 'max']
}

params_pyod_isolation_forest = {
    "model__contamination": [DEFAULT_CONTAMINATION],
    "model__n_estimators": [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    "model__max_samples": [0.5, 0.7, 1.0, 'auto'],
    "model__max_features": [0.5, 0.7, 1.0]
}

params_pyod_loda = {
    "model__contamination": [DEFAULT_CONTAMINATION],
    "model__n_bins": ['auto'],
    "model__n_random_cuts": [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
}

params_pyod_hbos = {
    "model__contamination": [DEFAULT_CONTAMINATION],
    "model__n_bins": ['auto'],
    "model__alpha": [0.1, 0.3, 0.5, 0.7, 0.9],
    "model__tol": [0.1, 0.3, 0.5, 0.7, 0.9]
}

params_pyod_sod = {
    "model__contamination": [DEFAULT_CONTAMINATION],
    "model__alpha": [0.1, 0.3, 0.5, 0.7, 0.9],
    "model__n_neighbors": [10, 15, 20, 25, 30]
}

params_pyod_xgbod = {
    # do not know, just default params were utilized
}

params_pyod_auto_encoder = {
    # do not know
}

params_pyod_cblof = {
    "model__contamination": [DEFAULT_CONTAMINATION],
    "model__n_clusters": [2, 3, 5, 8, 10, 15],
    "model__alpha": [0.6, 0.7, 0.8, 0.9],
    "model__beta": [2, 5, 10, 15, 20, 50, 100],
    "model__use_weights": [True, False]
}

params_pyod_copod = {
    "model__contamination": [DEFAULT_CONTAMINATION]
}

params_pyod_cof = {
    "model__contamination": [DEFAULT_CONTAMINATION],
    "model__n_neighbors": [3, 5, 7, 10, 15, 20],
    "model__method": ['fast', 'memory']
}

params_pyod_knn = {
    "model__contamination": [DEFAULT_CONTAMINATION],
    "model__n_neighbors": [3, 5, 7, 10],
    "model__method": ['largest', 'mean', 'median'],
    "model__algorithm": ['auto']
}

params_pyod_lmdd = {
    "model__contamination": [DEFAULT_CONTAMINATION],
    "model__n_iter": [20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    "model__dis_measure": ['aad', 'var', 'iqr']
}

params_pyod_loci = {
    "model__contamination": [DEFAULT_CONTAMINATION],
    "model__alpha": [0.5, 1.0, 2.0, 2.5, 5.0],
    "model__k": [1, 3, 5, 7, 10]
}

params_pyod_lof = {
    "model__contamination": [DEFAULT_CONTAMINATION],
    "model__n_neighbors": [3, 5, 7, 10, 15, 20],
    "model__algorithm": ['ball_tree', 'kd_tree', 'brute', 'auto'],
    "model__metric": ["euclidean", "minkowski"]
}

params_pyod_mcd = {
    "model__contamination": [DEFAULT_CONTAMINATION],
    "model__assume_centered": [True, False]
}

params_pyod_sos = {
    "model__contamination": [DEFAULT_CONTAMINATION],
    "model__perplexity": [0.001, 0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 7.0, 10.0],
    "model__metric": ["euclidean", "minkowski"],
    "model__eps": [0.001, 0.01, 0.1, 0.5, 1.0]
}

params_rus_boost = {
    "model__n_estimators": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    "model__learning_rate": [0.001, 0.01, 0.1, 0.5, 1, 3, 5, 10, 15, 20],
    "model__sampling_strategy": ['not minority', 'majority', 'auto'],
}

params_adaboost = {
    "model__n_estimators": [10, 50, 100, 150, 200, 250, 300, 350, 400, 450]
}

params_random_forest = {
    "model__n_estimators": [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    "model__max_depth": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    "model__max_features": ['auto', 'log2', 5, 10, 15, 20],
    "model__criterion": ['gini', 'entropy']
}

params_easy_ensemble = {
    "model__n_estimators": [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
}

params_balanced_random_forest = {
    "model__n_estimators": [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    "model__criterion": ['gini', 'entropy'],
    "model__max_features": ['auto', 'log2', 5, 10, 15, 20],
    "model__sampling_strategy": ['not minority', 'majority', 'auto']
}

params_balanced_bagging = {
    "model__n_estimators": [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    "model__sampling_strategy": ['not minority', 'majority', 'auto'],
    "model__max_features": [5, 10, 15, 20]
}

params_one_class_svm = {
    "model__degree": [1, 2, 3],
    "model__nu": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "model__gamma": [0.01, 0.1, 1, 3, 5]
}

params_least_squares_anomaly_detection = {
    "model__rho": [0.001, 0.01, 0.1, 0.5, 1, 3, 5, 10],
    "model__sigma": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5, 10]
}

params_isolation_forest = {
    "model__n_estimators": [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    "model__contamination": [0.1, 0.2, 0.3, 0.4, 0.5],
    "model__max_samples": [256, 512, 1024, 2048, 'auto']
}

params_imbalanced_ensemble_balanced_cascade = {
    "model__n_estimators": [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    "model__estimator__criterion": ['gini', 'entropy'],
    "model__estimator__splitter": ['best', 'random'],
    "model__estimator__ccp_alpha": [0.1, 0.3, 0.5, 0.7, 0.9]
}

params_imbalanced_ensemble_self_paced = {
    "model__n_estimators": [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    "model__estimator_params__criterion": ['gini', 'entropy'],
    "model__estimator_params__splitter": ['best', 'random'],
    "model__estimator_params__ccp_alpha": [0.1, 0.3, 0.5, 0.7, 0.9]
}

params_imbalanced_ensemble_balanced_random_forest = {
    "model__n_estimators": [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    "model__criterion": ['gini', 'entropy'],
    "model__max_features": ['auto', 'log2', 5, 10, 15, 20],
    "model__ccp_alpha": [0.1, 0.3, 0.5, 0.7, 0.9]
}

params_imbalanced_ensemble_easy_ensemble = {
    "model__n_estimators": [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    "model__max_features": ['auto', 'log2', 5, 10, 15, 20],
    "model__warm_start": [True, False]
}

params_imbalanced_ensemble_underbagging = {
    "model__n_estimators": [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    "model__max_features": ['auto', 'log2', 5, 10, 15, 20],
    "model__warm_start": [True, False]
}

params_imbalanced_ensemble_overboost = {
    "model__n_estimators": [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    "model__learning_rate": [0.001, 0.01, 0.1, 1],
    "model__algorithm": ['SAMME', 'SAMME.R']
}

params_imbalanced_ensemble_smote_boost = {
    "model__n_estimators": [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    "model__learning_rate": [0.001, 0.01, 0.1, 1],
    "model__k_neighbors": [3, 4, 5],
    "model__algorithm": ['SAMME', 'SAMME.R']
}

params_imbalanced_ensemble_smote_bagging = {
    "model__n_estimators": [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    "model__max_features": [5, 10, 15, 20],
    "model__k_neighbors": [3, 4, 5],
    "model__warm_start": [True, False]
}

params_imbalanced_ensemble_kmeans_smote_boost = {
    "model__n_estimators": [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    "model__learning_rate": [0.001, 0.01, 0.1, 1],
    "model__k_neighbors": [3, 4, 5],
    "model__cluster_balance_threshold": ['auto', 0.1, 0.01, 0.001, 0.0001,0.00005,0.00001],
    "model__algorithm": ['SAMME', 'SAMME.R']
}

params_imbalanced_ensemble_adacost = {
    "model__n_estimators": [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    "model__learning_rate": [0.001, 0.01, 0.1, 1],
    "model__algorithm": ['SAMME', 'SAMME.R']
}

params_imbalanced_ensemble_adauboost = {
    "model__n_estimators": [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    "model__learning_rate": [0.001, 0.01, 0.1, 1],
    "model__algorithm": ['SAMME', 'SAMME.R']
}

params_svc = {
    "model__gamma": [0.001, 0.01, 0.1, 1, 10, 100],
    "model__C": [0.01, 0.1, 1, 5, 10, 100]
}

params_weighted_svc = {
    "model__gamma": [0.001, 0.01, 0.1, 1, 10, 100],
    "model__C": [0.01, 0.1, 1, 5, 10, 100],
    "model__class_weight": ['balanced', {0: 100, 1: 1}, {0: 75, 1: 1}, {0: 50, 1: 1}, {0: 25, 1: 1}, {0: 10, 1: 1},
                            {0: 1, 1: 1}, {0: 1, 1: 10}, {0: 1, 1: 25}, {0: 1, 1: 50}, {0: 1, 1: 75}, {0: 1, 1: 100}]
}

params_weighted_random_forest = {
    "model__n_estimators": [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    "model__max_depth": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    "model__max_features": ['auto', 'log2', 5, 10, 15, 20],
    "model__criterion": ['gini', 'entropy'],
    "model__class_weight": ['balanced', {0: 100, 1: 1}, {0: 75, 1: 1}, {0: 50, 1: 1}, {0: 25, 1: 1}, {0: 10, 1: 1},
                            {0: 1, 1: 1}, {0: 1, 1: 10}, {0: 1, 1: 25}, {0: 1, 1: 50}, {0: 1, 1: 75}, {0: 1, 1: 100}]
}

params_xgboost = {
    "model__gamma": [0.001, 0.01, 0.1, 1, 10, 100],
    "model__learning_rate": [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9],
    "model__max_depth": [3, 5, 10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    "model__min_child_weight": [1, 3, 5, 10],
    "model__subsample": [0.8, 0.9, 1.0]
}