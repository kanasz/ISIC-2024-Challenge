from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer

from tabular_tuning.tabular_tuning_constants import SEED, NUMERICAL_FEATURE, CATEGORICAL_FEATURES
from imblearn.ensemble import RUSBoostClassifier, EasyEnsembleClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
from imblearn.metrics import geometric_mean_score, sensitivity_score, specificity_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, IsolationForest
from imblearn.combine import SMOTETomek, SMOTEENN
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.copod import COPOD
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.lmdd import LMDD
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.loci import LOCI
from pyod.models.mcd import MCD
from pyod.models.pca import PCA
from pyod.models.sod import SOD
from pyod.models.sos import SOS
from pyod.models.xgbod import XGBOD
from pyod.models.iforest import IForest
from pyod.models.feature_bagging import FeatureBagging
from sklearn.svm import OneClassSVM, SVC
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SMOTEN
from imblearn.under_sampling import NearMiss, TomekLinks, ClusterCentroids, RandomUnderSampler
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
#from base_constants import ClassifierType, Classifier, DataOrigin, RANDOM_STATE, N_ITERATIONS, SearchType
from xgboost import XGBClassifier
from pathlib import Path
from base_constants import MetricType, Classifier, RANDOM_STATE, SearchType, ClassifierType, N_ITERATIONS

import math
import warnings
import pandas as pd
import csv
import os
import glob
import re
import imbalanced_ensemble.ensemble as imb

from tabular_tuning.tabular_tuning_functions import get_preprocessor, custom_pauc

warnings.filterwarnings('ignore')

custom_scorer = make_scorer(custom_pauc, needs_proba=True, greater_is_better=True)
def get_scoring_dict():
    scoring_dict = {
        'accuracy_score': make_scorer(accuracy_score),
        'f1_score': make_scorer(f1_score),
        'roc_auc_score': make_scorer(roc_auc_score),
        'geometric_mean_score': make_scorer(geometric_mean_score),
        'sensitivity_score': make_scorer(sensitivity_score),
        'specificity_score': make_scorer(specificity_score),
        'pauc_score': custom_scorer
    }
    return scoring_dict


# get pipeline for RandomizedSearchCV with selected classifier
def get_pipeline(classifier_name, use_one_hot = True):
    if classifier_name == Classifier.PYOD_IF:
        clf = IForest(random_state=RANDOM_STATE)
    elif classifier_name == Classifier.PYOD_CBLOF:
        clf = CBLOF(random_state=RANDOM_STATE)
    elif classifier_name == Classifier.PYOD_COPOD:
        clf = COPOD()
    elif classifier_name == Classifier.PYOD_COF:
        clf = COF()
    elif classifier_name == Classifier.PYOD_HBOS:
        clf = HBOS()
    elif classifier_name == Classifier.PYOD_KNN:
        clf = KNN()
    elif classifier_name == Classifier.PYOD_LMDD:
        clf = LMDD(random_state=RANDOM_STATE)
    elif classifier_name == Classifier.PYOD_LODA:
        clf = LODA()
    elif classifier_name == Classifier.PYOD_LOF:
        clf = LOF()
    elif classifier_name == Classifier.PYOD_LOCI:
        clf = LOCI()
    elif classifier_name == Classifier.PYOD_MCD:
        clf = MCD(random_state=RANDOM_STATE)
    elif classifier_name == Classifier.PYOD_PCA:
        clf = PCA(random_state=RANDOM_STATE)
    elif classifier_name == Classifier.PYOD_SOD:
        clf = SOD(n_neighbors=20)
    elif classifier_name == Classifier.PYOD_SOS:
        clf = SOS()
    elif classifier_name == Classifier.PYOD_XGBOD:
        clf = XGBOD(random_state=RANDOM_STATE)
    elif classifier_name == Classifier.PYOD_FEATURE_BAGGING:
        clf = FeatureBagging(random_state=RANDOM_STATE)
    elif classifier_name == Classifier.ADABOOST:
        clf = AdaBoostClassifier(random_state=RANDOM_STATE)
    elif classifier_name == Classifier.RANDOM_FOREST:
        clf = RandomForestClassifier(random_state=RANDOM_STATE)
    elif classifier_name == Classifier.RUS_BOOST:
        clf = RUSBoostClassifier(random_state=RANDOM_STATE)
    elif classifier_name == Classifier.EASY_ENSEMBLE:
        clf = EasyEnsembleClassifier(random_state=RANDOM_STATE)
    elif classifier_name == Classifier.BALANCED_RANDOM_FOREST:
        clf = BalancedRandomForestClassifier(random_state=RANDOM_STATE)
    elif classifier_name == Classifier.BALANCED_BAGGING:
        clf = BalancedBaggingClassifier(random_state=RANDOM_STATE)
    #elif classifier_name == Classifier.LEAST_SQUARES_ANOMALY_DETECTION:
    #    clf = LSAnomaly()
    elif classifier_name == Classifier.ISOLATION_FOREST:
        clf = IsolationForest(random_state=RANDOM_STATE)
    elif classifier_name == Classifier.ONE_CLASS_SVM:
        clf = OneClassSVM()
    elif classifier_name == Classifier.IMBALANCED_BALANCED_CASCADE:
        clf = imb.BalanceCascadeClassifier(estimator=DecisionTreeClassifier(), random_state=RANDOM_STATE)
    elif classifier_name == Classifier.IMBALANCED_SELF_PACED:
        clf = imb.SelfPacedEnsembleClassifier(estimator=DecisionTreeClassifier(), random_state=RANDOM_STATE)
    elif classifier_name == Classifier.IMBALANCED_BALANCED_RANDOM_FOREST:
        clf = imb.BalancedRandomForestClassifier(random_state=RANDOM_STATE)
    elif classifier_name == Classifier.IMBALANCED_EASY_ENSEMBLE:
        clf = imb.EasyEnsembleClassifier(random_state=RANDOM_STATE, estimator=AdaBoostClassifier(n_estimators=30))
    elif classifier_name == Classifier.IMBALANCED_UNDERBAGGING:
        clf = imb.UnderBaggingClassifier(random_state=RANDOM_STATE)
    elif classifier_name == Classifier.IMBALANCED_OVERBOOST:
        clf = imb.OverBoostClassifier(random_state=RANDOM_STATE)
    elif classifier_name == Classifier.WEIGHTED_SVC:
        clf = SVC(random_state=RANDOM_STATE)
    elif classifier_name == Classifier.WEIGHTED_RANDOM_FOREST:
        clf = RandomForestClassifier(random_state=RANDOM_STATE)
    elif classifier_name == Classifier.IMBALANCED_SMOTE_BOOST:
        clf = imb.SMOTEBoostClassifier(random_state=RANDOM_STATE, k_neighbors=5)
    elif classifier_name == Classifier.IMBALANCED_KMEANS_SMOTE_BOOST:
        clf = imb.KmeansSMOTEBoostClassifier(random_state=RANDOM_STATE)
    elif classifier_name == Classifier.IMBALANCED_SMOTE_BAGGING:
        clf = imb.SMOTEBaggingClassifier(random_state=RANDOM_STATE, k_neighbors=5)
    elif classifier_name == Classifier.IMBALANCED_ADACOST:
        clf = imb.AdaCostClassifier(random_state=RANDOM_STATE)
    elif classifier_name == Classifier.IMBALANCED_ADAUBOOST:
        clf = imb.AdaUBoostClassifier(random_state=RANDOM_STATE)
    else:
        raise Exception("Unsupported classifier!")


    # Numerical pipeline: Impute missing values then scale
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: Impute missing values then one-hot encode
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),

    ])

    if use_one_hot:
        categorical_pipeline.steps.append(('onehot', OneHotEncoder(handle_unknown='ignore')))

    # Combine the numerical and categorical pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, NUMERICAL_FEATURE),
            ('cat', categorical_pipeline, CATEGORICAL_FEATURES)
        ]
    )

    # Create the full pipeline by adding the model at the end
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        #('brf_undersample', RandomUnderSampler(sampling_strategy=0.01, random_state=SEED)),
        ('model', clf)
    ])

    return model_pipeline


# get pipeline with sampling for selected classifier
def get_pipeline_with_sampling(classifier_name, k_neighbors, use_one_hot=True):

    if classifier_name == Classifier.SAMPLING_SMOTE_SVC:
        sampler = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
        clf = SVC(random_state=RANDOM_STATE)
    elif classifier_name == Classifier.SAMPLING_ADASYN_SVC:
        sampler = ADASYN(random_state=RANDOM_STATE, n_neighbors=k_neighbors)
        clf = SVC(random_state=RANDOM_STATE)
    elif classifier_name == Classifier.SAMPLING_BORDERLINESMOTE_SVC:
        sampler = BorderlineSMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
        clf = SVC(random_state=RANDOM_STATE)
    elif classifier_name == Classifier.SAMPLING_SMOTEN_SVC:
        sampler = SMOTEN(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
        clf = SVC(random_state=RANDOM_STATE)
    elif classifier_name == Classifier.SAMPLING_NEAR_MISS_SVC:
        sampler = NearMiss(n_neighbors=k_neighbors, sampling_strategy='majority')
        clf = SVC(random_state=RANDOM_STATE)
    elif classifier_name == Classifier.SAMPLING_TOMEK_LINKS_SVC:
        sampler = TomekLinks(sampling_strategy='majority')
        clf = SVC(random_state=RANDOM_STATE)
    elif classifier_name == Classifier.SAMPLING_CLUSTER_CENTROIDS_SVC:
        sampler = ClusterCentroids(sampling_strategy='majority')
        clf = SVC(random_state=RANDOM_STATE)
    elif classifier_name == Classifier.SAMPLING_COMBINE_SMOTE_TOMEK_SVC:
        sampler = SMOTETomek(sampling_strategy='all')
        clf = SVC(random_state=RANDOM_STATE)
    elif classifier_name == Classifier.SAMPLING_COMBINE_SMOTE_ENN_SVC:
        sampler = SMOTEENN(sampling_strategy='all')
        clf = SVC(random_state=RANDOM_STATE)
    # XGBOOST
    elif classifier_name == Classifier.SAMPLING_SMOTE_XGBOOST:
        sampler = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
        clf = XGBClassifier()
    elif classifier_name == Classifier.SAMPLING_ADASYN_XGBOOST:
        sampler = ADASYN(random_state=RANDOM_STATE, n_neighbors=k_neighbors)
        clf = XGBClassifier()
    elif classifier_name == Classifier.SAMPLING_BORDERLINESMOTE_XGBOOST:
        sampler = BorderlineSMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
        clf = XGBClassifier()
    elif classifier_name == Classifier.SAMPLING_SMOTEN_XGBOOST:
        sampler = SMOTEN(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
        clf = XGBClassifier()
    elif classifier_name == Classifier.SAMPLING_NEAR_MISS_XGBOOST:
        sampler = NearMiss(n_neighbors=k_neighbors, sampling_strategy='majority')
        clf = XGBClassifier()
    elif classifier_name == Classifier.SAMPLING_TOMEK_LINKS_XGBOOST:
        sampler = TomekLinks(sampling_strategy='majority')
        clf = XGBClassifier()
    elif classifier_name == Classifier.SAMPLING_CLUSTER_CENTROIDS_XGBOOST:
        sampler = ClusterCentroids(sampling_strategy='majority')
        clf = XGBClassifier()
    elif classifier_name == Classifier.SAMPLING_COMBINE_SMOTE_TOMEK_XGBOOST:
        sampler = SMOTETomek(sampling_strategy='all')
        clf = XGBClassifier()
    elif classifier_name == Classifier.SAMPLING_COMBINE_SMOTE_ENN_XGBOOST:
        sampler = SMOTEENN(sampling_strategy='all')
        clf = XGBClassifier()
    else:
        raise Exception("Unsupported classifier!")


    # Numerical pipeline: Impute missing values then scale
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: Impute missing values then one-hot encode
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),

    ])

    if use_one_hot:
        categorical_pipeline.steps.append(('onehot', OneHotEncoder(handle_unknown='ignore')))

    # Combine the numerical and categorical pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, NUMERICAL_FEATURE),
            ('cat', categorical_pipeline, CATEGORICAL_FEATURES)
        ]
    )

    # Create the full pipeline by adding the model at the end
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ("sampling", sampler),
        ('model', clf)
    ])

    return model_pipeline


# do prediction with selected data and classifier
# *args - optional parameter for number of neighbors for sampling algorithms
def do_prediction(data, labels, groups, classifier_name, classifier_type, params_dict, search_type=SearchType.GRID_SEARCH,
                  sampling_k_neighbors=5, use_one_hot = True):
    warnings.filterwarnings('ignore')
    print("sampling neigh: {}".format(sampling_k_neighbors))
    if classifier_name == Classifier.ONE_CLASS_SVM:
        labels = labels.apply(lambda x: -1 if x == 0 else 1)
    elif classifier_name == Classifier.ISOLATION_FOREST:
        labels = labels.apply(lambda x: -1 if x == 0 else 1)

    print(labels.shape[0])
    if classifier_type == ClassifierType.SEMI_SUPERVISED:
        print("Using semi-supervised classifier")
        #k_fold = ImbalancedKFold(n_splits=5)
        k_fold = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    elif classifier_type == ClassifierType.SUPERVISED:
        print("Using supervised classifier")
        k_fold = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    else:
        raise Exception("Unsupported operation!")

    print("###########################################################")
    print("Running...")

    # with sampling
    if any([classifier_name == Classifier.SAMPLING_SMOTE_SVC,
            classifier_name == Classifier.SAMPLING_SMOTEN_SVC,
            classifier_name == Classifier.SAMPLING_ADASYN_SVC,
            classifier_name == Classifier.SAMPLING_BORDERLINESMOTE_SVC,
            classifier_name == Classifier.SAMPLING_NEAR_MISS_SVC,
            classifier_name == Classifier.SAMPLING_TOMEK_LINKS_SVC,
            classifier_name == Classifier.SAMPLING_CLUSTER_CENTROIDS_SVC,
            classifier_name == Classifier.SAMPLING_COMBINE_SMOTE_TOMEK_SVC,
            classifier_name == Classifier.SAMPLING_COMBINE_SMOTE_ENN_SVC,
            classifier_name == Classifier.SAMPLING_SMOTE_XGBOOST,
            classifier_name == Classifier.SAMPLING_SMOTEN_XGBOOST,
            classifier_name == Classifier.SAMPLING_BORDERLINESMOTE_XGBOOST,
            classifier_name == Classifier.SAMPLING_ADASYN_XGBOOST,
            classifier_name == Classifier.SAMPLING_CLUSTER_CENTROIDS_XGBOOST,
            classifier_name == Classifier.SAMPLING_NEAR_MISS_XGBOOST,
            classifier_name == Classifier.SAMPLING_TOMEK_LINKS_XGBOOST,
            classifier_name == Classifier.SAMPLING_COMBINE_SMOTE_TOMEK_XGBOOST,
            classifier_name == Classifier.SAMPLING_COMBINE_SMOTE_ENN_XGBOOST]):
        grid_search_instance = GridSearchCV(get_pipeline_with_sampling(classifier_name, sampling_k_neighbors, use_one_hot),
                                            params_dict, n_jobs=20, return_train_score=True, verbose=1,
                                            refit='geometric_mean_score', scoring=get_scoring_dict(), cv=k_fold)
        grid_search_instance.fit(data, labels, groups=groups)
        print("Prediction task is completed!")
        print("###########################################################")
        return grid_search_instance.best_params_, grid_search_instance.cv_results_

    # without sampling
    else:
        if search_type == SearchType.RANDOM_SEARCH:
            randomized_search_instance = RandomizedSearchCV(get_pipeline(classifier_name, use_one_hot), params_dict, verbose=1,
                                                            n_iter=N_ITERATIONS, random_state=RANDOM_STATE, n_jobs=15,
                                                            return_train_score=True, refit="pauc_score",
                                                            scoring=get_scoring_dict(), cv=k_fold)
        else:
            randomized_search_instance = GridSearchCV(get_pipeline(classifier_name, use_one_hot), params_dict, n_jobs=15, verbose=2,
                                                      return_train_score=True, refit="pauc_score",
                                                      scoring=get_scoring_dict(), cv=k_fold)
        randomized_search_instance.fit(data, labels, groups=groups)
        print("Prediction task is completed!")
        print("###########################################################")
        return randomized_search_instance.best_params_, randomized_search_instance.cv_results_


# save prediction results to csv file
def save_results(clf, file_name, best_params, cv_results):
    print("Saving prediction results...")
    df = pd.DataFrame()
    for param in cv_results:
        if param.find('train') > -1:
            continue
        df[param] = cv_results[param]

    columns_to_drop = ['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time', 'params']
    df = df.drop(columns=columns_to_drop)
    path = Path(__file__).parent / "predictions/{}/results/".format(clf)
    if not os.path.exists(path):
        os.makedirs(path)

    df.to_csv(os.path.join(path, file_name + "_results.csv"))
    path = Path(__file__).parent / "predictions/{}/results/{}_best_params.csv".format(clf, file_name)
    w = csv.writer(open(path, "w"))
    for key, val in best_params.items():
        w.writerow([key.replace('model__', ''), val])


# method to print best g-mean results if exists
'''
def print_results(clf_type, metric_type: MetricType, append_auc_to_g_mean: bool, list_of_clf_to_analyze: list, append_miss_rate_to_gmean=False):
    """
    This method prints the best results according to selected metric
    :param list_of_clf_to_analyze: expected list
    :param clf_type: selected classifier or all supported classifiers
    :param metric_type: type of metric, that will be summarized
    :param append_auc_to_g_mean: should method provide results for UAC score metric according to the best GM scores
    :return: dataframe with results
    :list_of_clf_to_analyze: list of classifiers that will be analyzed
    """
    path = Path(__file__).parent / "predictions"
    os.chdir(path)
    print(os.getcwd())
    results = {
        "Clf": [],
        "Dataset": [],
        metric_type.value: []
    }
    # only if needed: append_auc_to_g_mean = True
    auc_related_results = {
        "Clf": [],
        "Dataset": [],
        "roc_auc_related_score": [],
        "miss_rate_related_score": []
    }

    # I want to print results of all classifiers
    if clf_type == Classifier.ALL:
        # iterate over predicted files
        for clf in Classifier:
            if clf in list_of_clf_to_analyze:
                print("Processing {} results...".format(clf.value))
                print("-----------------------------------------------------------------------------------")
                path = os.getcwd() + '/{}/results'.format(clf.value)
                if os.path.exists(path):
                    os.chdir('./{}/results'.format(clf.value))
                    for data_origin in DataOrigin:
                        if os.path.exists(os.getcwd() + '/{}'.format(data_origin.value)):
                            os.chdir('./{}'.format(data_origin.value))
                            # collect files with postfix "_results.csv" and filter out agriculture and retail
                            result_files = [
                                f for f in glob.glob('{}/*_results.csv'.format(os.getcwd()))
                                if not re.search(r"(agriculture|retail)", f, re.IGNORECASE)
                            ]

                            # verify if any files were found
                            if not result_files:
                                print("No files were found!")
                            else:
                                for file in result_files:
                                    tmp_results = pd.read_csv("{}".format(file))
                                    max_index = tmp_results["mean_test_" + metric_type.value + "_score"].idxmax()
                                    if math.isnan(max_index):
                                        continue
                                    max_row = pd.DataFrame(tmp_results.iloc[[max_index]])
                                    g_mean_best = max_row['mean_test_' + metric_type.value + '_score'].iloc[0]
                                    g_mean_std = max_row['std_test_' + metric_type.value + '_score'].iloc[0]
                                    print("DATA: {}".format(file))
                                    print(f"CLF: {clf.value}, {metric_type.value.upper()}: {g_mean_best * 100:.2f} ±"
                                          f" {g_mean_std * 100:.2f}")
                                    ds = os.path.basename(file.replace('_results.csv', ''))
                                    if 'agriculture' not in ds and 'retail' not in ds:
                                        # append G-MEAN results
                                        results['Clf'].append(clf.value)
                                        results[metric_type.value].append(g_mean_best)
                                        results['Dataset'].append(ds)
                                        # append ROC AUC score results
                                        if append_auc_to_g_mean:
                                            roc_auc = max_row['mean_test_roc_auc_score'].iloc[0]
                                            roc_auc_std = max_row['std_test_roc_auc_score'].iloc[0]
                                            print(
                                                f"CLF: {clf.value}, {'roc auc'.upper()}: "
                                                f"{roc_auc * 100:.2f} ± {roc_auc_std * 100:.2f}")
                                            auc_related_results['Clf'].append(clf.value)
                                            auc_related_results['roc_auc_related_score'].append(roc_auc)
                                            auc_related_results['Dataset'].append(ds)
                                        if append_miss_rate_to_gmean:
                                            miss_rate = 1-max_row['mean_test_sensitivity_score'].iloc[0]
                                            miss_rate_std = max_row['std_test_sensitivity_score'].iloc[0]
                                            print(
                                                f"CLF: {clf.value}, {'miss_rate'.upper()}: "
                                                f"{miss_rate * 100:.2f} ± {miss_rate_std * 100:.2f}")
                                            #auc_related_results['Clf'].append(clf.value)
                                            auc_related_results['miss_rate_related_score'].append(f"{miss_rate * 100:.1f} ± {miss_rate_std * 100:.0f}")
                                            #auc_related_results['Dataset'].append(ds)
                                    print("---------------------------------------------------------------------------")

                            # return to "./results" directory
                            os.chdir('..')
                        else:
                            print("Results for data {} are missing!".format(data_origin.value))

                    # return to "./predictions" directory
                    os.chdir('../..')
                else:
                    print("Skipping... results are missing")
                print('###############################################################################################')

    elif isinstance(clf_type, Classifier) and not clf_type == Classifier.ALL:
        print("Processing {} results...".format(clf_type.value))
        print("-----------------------------------------------------------------------------------")
        path = os.getcwd() + '/{}/results'.format(clf_type.value)
        if os.path.exists(path):
            os.chdir('./{}/results'.format(clf_type.value))
            for data_origin in DataOrigin:
                if os.path.exists(os.getcwd() + '/{}'.format(data_origin.value)):
                    os.chdir('./{}'.format(data_origin.value))
                    # collect files with postfix "_results.csv"
                    result_files = glob.glob('{}/*_results.csv'.format(os.getcwd()))
                    # verify if any files were found
                    if not result_files:
                        print("No files were found!")
                    else:
                        for file in result_files:
                            tmp_results = pd.read_csv("{}".format(file))
                            max_index = tmp_results["mean_test_geometric_mean_score"].idxmax()
                            if math.isnan(max_index):
                                continue
                            max_row = pd.DataFrame(tmp_results.iloc[[max_index]])
                            g_mean_best = max_row['mean_test_' + metric_type.value + '_score'].iloc[0]
                            g_mean_std = max_row['std_test_' + metric_type.value + '_score'].iloc[0]
                            print("DATA: {}".format(file))
                            print("CLF: {}, G_MEAN: {:.2f} ± {:.2f}".format(clf_type.value, g_mean_best * 100,
                                                                            g_mean_std * 100))
                            print("-------------------------------------------------------------------------------")

                    # return to "./results" directory
                    os.chdir('..')
                else:
                    print("Results for data {} are missing!".format(data_origin.value))

            # return to "./predictions" directory
            os.chdir('../..')
        else:
            print("Skipping... results are missing")
    else:
        print("Unsupported operation!")

    if append_auc_to_g_mean:
        return pd.DataFrame(results), pd.DataFrame(auc_related_results)
    else:
        return pd.DataFrame(results)
'''