import numpy as np
import pygad
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from ga_tabular_tuning.ga_tuning_functions import custom_metric
from tabular_tuning.tabular_tuning_constants import CATEGORICAL_FEATURES
from tabular_tuning.tabular_tuning_functions import get_data, get_preprocessor
import time

sampling_ratio = 0.01
SEED = 42
FILENAME = "ga_tuning_solution"
data, labels, groups = get_data()
X = data
y = labels

def fitness_func(ga_instance, solution, solution_idx):
    start_time = time.time()
    xgb_params = {
        'random_state': SEED,
        'n_estimators': int(solution[0]),
        'max_depth': int(solution[1]),
        'learning_rate': solution[2],
        'subsample': solution[3],
        'colsample_bytree': solution[4],
        'gamma': solution[5],
        'min_child_weight': solution[6],
        'scale_pos_weight': solution[7],
        'tree_method': ['auto', 'exact', 'approx', 'hist', 'gpu_hist'][int(solution[8])],
        'alpha': solution[9],
        'lambda': solution[10],
        'colsample_bylevel': solution[11]#,
        #'colsample_bynode': solution[12]
    }

    lgb_params = {
        'n_estimators': int(solution[13]),
        'max_depth': int(solution[14]),
        'learning_rate': solution[15],
        'subsample': solution[16],
        'colsample_bytree': solution[17],
        'min_child_weight': solution[18],
        'scale_pos_weight': solution[19],
        'num_leaves': int(solution[20]),
        'objective': 'binary', #[int(solution[21])],
        'boosting_type': ['gbdt', 'rf', 'dart', 'goss'][int(solution[22])],
        'lambda_l1': solution[23],
        'lambda_l2': solution[24],
        'colsample_bynode': solution[25],
        #'bagging_fraction': solution[26],
        #'bagging_freq': int(solution[27]),
        'min_data_in_leaf': int(solution[28]),
        'n_iter': int(solution[29]),
        #'boosting_type': ['gbdt', 'rf', 'dart', 'goss'][int(solution[30])],
        'verbosity':-1,
        'force_col_wise':True,
        'random_state':SEED
    }

    cat_features = [X.columns.get_loc(col) for col in CATEGORICAL_FEATURES]

    cat_params = {
        #'loss_function': ['Logloss', 'CrossEntropy', 'RMSE', 'MultiClass', 'Quantile'][int(solution[31])],
        'loss_function': 'Logloss',
        'iterations': int(solution[32]),
        'verbose': False,
        'random_state': SEED,
        'max_depth': int(solution[33]),
        'learning_rate': solution[34],
        'scale_pos_weight': solution[35],
        'l2_leaf_reg': solution[36],
        'subsample': solution[37],
        'min_data_in_leaf': int(solution[38]),
        'cat_features': cat_features,


    }



    cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)

    xgb_model = Pipeline([
        ('brf_preprocessor', get_preprocessor(True)),
        ('sampler_1', RandomOverSampler(sampling_strategy=0.003, random_state=SEED)),
        ('sampler_2', RandomUnderSampler(sampling_strategy=sampling_ratio, random_state=SEED)),
        ('classifier', XGBClassifier(**xgb_params)),
    ])

    lgb_model = Pipeline([
        ('brf_preprocessor', get_preprocessor(True)),
        ('sampler_1', RandomOverSampler(sampling_strategy=0.003, random_state=SEED)),
        ('sampler_2', RandomUnderSampler(sampling_strategy=sampling_ratio, random_state=SEED)),
        ('classifier', LGBMClassifier(**lgb_params)),
    ])

    cb_model = Pipeline([
        ('brf_preprocessor', get_preprocessor(False)),
        ('sampler_1', RandomOverSampler(sampling_strategy=0.003, random_state=SEED)),
        ('sampler_2', RandomUnderSampler(sampling_strategy=sampling_ratio, random_state=SEED)),
        ('classifier', CatBoostClassifier(**cat_params)),
    ])

    estimator = VotingClassifier([
        ('cb', cb_model), ('xgb', xgb_model), ('lgb', lgb_model)
    ], voting='soft')

    val_score = cross_val_score(
        n_jobs=5,
        estimator=estimator,
        X=X, y=y,
        cv=cv,
        groups=groups,
        scoring=custom_metric,
    )

    fitness = np.mean(val_score)
    t = time.time() - start_time
    print("pAUC: {0:.10f}, {1:.25f} seconds".format(fitness, t))
    return fitness

ga_instance = pygad.load("ga_tabular_tuning/ga_tuning_solution")
best_solution = ga_instance.best_solutions[-1]

fitness_func(None, best_solution, None)