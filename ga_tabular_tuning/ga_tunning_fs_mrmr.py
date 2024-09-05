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
import concurrent.futures
import warnings
from utils.mrmrfeatureselector import MRMRFeatureSelector

warnings.simplefilter("ignore")

sampling_ratio = 0.01
SEED = 42
FILENAME = "ga_tabular_tuning/ga_tuning_mrmr_solution"
data, labels, groups = get_data()
X = data
y = labels


# Function to handle each fold's operations
def process_fold(fold_data, n_features, xgb_params, lgb_params, cat_params, other_params):
    fold, (train_idx, test_idx) = fold_data



    # Split data into training and test sets
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    group_train, group_test = groups.iloc[train_idx], groups.iloc[test_idx]

    # Step 1: Apply oversampling and undersampling only to the training set
    ros = RandomOverSampler(sampling_strategy=other_params["oversampling_strategy"], random_state=42)
    rus = RandomUnderSampler(sampling_strategy=other_params["undersampling_strategy"], random_state=42)  # Example ratio
    #fs = MRMRFeatureSelector(n_features=n_features)
    preprocessor = get_preprocessor(True)

    # Preprocess the data
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    #X_train = fs.fit_transform(X_train, y_train.values)
    #X_test = fs.transform(X_test)

    # Resample the training data
    try:
        X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
        X_train_resampled, y_train_resampled = rus.fit_resample(X_train_resampled, y_train_resampled)
    except:
        return 0
    # Step 2: Define models
    xgb_model = XGBClassifier(**xgb_params)
    lgb_model = LGBMClassifier(**lgb_params)
    cb_model = CatBoostClassifier(**cat_params)

    # Step 3: Voting classifier
    estimator = VotingClassifier([
        ('cb', cb_model),
        ('xgb', xgb_model),
        ('lgb', lgb_model)
    ], voting='soft')

    # Step 4: Fit the model on resampled training data
    estimator.fit(X_train_resampled, y_train_resampled)

    # Step 5: Evaluate on the untouched test set (validation set)
    score = custom_metric(estimator, X_test, y_test)

    return score

def fitness_func(ga_instance, solution, solution_idx):
    #data, labels, groups = get_data()
    #X = data
    #y = labels

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
        #'cat_features': cat_features,


    }

    n_features = int(solution[39])
    undersampling_strategy = solution[40]
    oversampling_strategy =  solution[41]

    other_params = {
        "undersampling_strategy":undersampling_strategy,
        "oversampling_strategy": oversampling_strategy
    }

    cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    # Use ThreadPoolExecutor to parallelize
    val_scores = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        fold_data = [(fold, split) for fold, split in enumerate(cv.split(X, y, groups=groups))]

        data = (fold_data, n_features, xgb_params, lgb_params, cat_params)
        # Map the process_fold function to each fold in parallel
        results = list(executor.map(lambda fd: process_fold(fd, n_features, xgb_params, lgb_params, cat_params, other_params), fold_data))

    # Collect results from parallel execution
    val_scores.extend(results)
    fitness = np.mean(val_scores)
    t = time.time() - start_time
    print("pAUC: {0:.10f}, {1:.25f} seconds".format(fitness, t))
    return fitness


def on_generation(ga_instance):
    print(f"Generation {ga_instance.generations_completed}: Best Fitness = {ga_instance.best_solutions_fitness[-1]}")
    print(f"Best solution found: {ga_instance.best_solutions[-1]}")
    ga_instance.save(filename=FILENAME)


def on_stop(ga_instance, last_population_fitness):
    print("GA has stopped running.")
    print(f"Best solution found: {ga_instance.best_solutions[-1]}")
    print(f"Fitness value of the best solution: {ga_instance.best_solutions_fitness[-1]}")


gene_space = [
    # XGB Parameters
    {'low': 10, 'high': 200},  # n_estimators (XGB)
    {'low': 1, 'high': 10},  # max_depth (XGB)
    {'low': 0.01, 'high': 0.3},  # learning_rate (XGB)
    {'low': 0.5, 'high': 1.0},  # subsample (XGB)
    {'low': 0.5, 'high': 1.0},  # colsample_bytree (XGB)
    {'low': 0, 'high': 10},  # gamma (XGB)
    {'low': 1, 'high': 10},  # min_child_weight (XGB)
    {'low': 1, 'high': 10},  # scale_pos_weight (XGB)
    {'low': 0, 'high': 4},  # tree_method (as an index to list) (XGB)
    {'low': 0, 'high': 10},  # alpha (L1 regularization term) (XGB)
    {'low': 0, 'high': 10},  # lambda (L2 regularization term) (XGB)
    {'low': 0.5, 'high': 1.0},  # colsample_bylevel (XGB)
    {'low': 0.5, 'high': 1.0},  # colsample_bynode (XGB)

    # LGBM Parameters
    {'low': 10, 'high': 200},  # n_estimators (LGBM)
    {'low': 1, 'high': 10},  # max_depth (LGBM)
    {'low': 0.01, 'high': 0.3},  # learning_rate (LGBM)
    {'low': 0.5, 'high': 1.0},  # subsample (LGBM)
    {'low': 0.5, 'high': 1.0},  # colsample_bytree (LGBM)
    {'low': 1, 'high': 10},  # min_child_weight (LGBM)
    {'low': 1, 'high': 10},  # scale_pos_weight (LGBM)
    {'low': 10, 'high': 120},  # num_leaves (LGBM)
    {'low': 0, 'high': 4},  # objective (as an index to list) (LGBM)
    {'low': 0, 'high': 4},  # boosting_type (as an index to list) (LGBM)
    {'low': 0, 'high': 10},  # lambda_l1 (LGBM)
    {'low': 0, 'high': 10},  # lambda_l2 (LGBM)
    {'low': 0.5, 'high': 1.0},  # colsample_bynode (LGBM)
    {'low': 0.5, 'high': 1.0},  # bagging_fraction (LGBM)
    {'low': 1, 'high': 10},  # bagging_freq (LGBM)
    {'low': 1, 'high': 100},  # min_data_in_leaf (LGBM)
    {'low': 50, 'high': 1000},  # n_iter (LGBM)
    {'low': 0, 'high': 4},  # boosting_type (as an index to list) (LGBM)

    # CatBoost Parameters
    {'low': 0, 'high': 4},          # loss_function (as an index to list)
    {'low': 100, 'high': 500},     # iterations
    {'low': 1, 'high': 10},         # max_depth
    {'low': 0.01, 'high': 0.3},     # learning_rate
    {'low': 1, 'high': 10},         # scale_pos_weight
    {'low': 1, 'high': 10},         # l2_leaf_reg
    {'low': 0.5, 'high': 1.0},      # subsample
    {'low': 1, 'high': 50}   ,        #


    {'low': 30, 'high': 70},         # n_features
    {'low': 0.005, 'high': 0.02},       #unsersampling strategy
    {'low': 0.001, 'high': 0.003}       #oversampling strategy
]

if __name__ == '__main__':
    # Initialize PyGAD
    ga_instance = pygad.GA(
        #save_best_solutions=True,
        parallel_processing=['process',3],
        save_best_solutions=True,
        random_seed=SEED,
        num_generations=100,  # Number of generations
        num_parents_mating=10,  # Number of parents for mating
        fitness_func=fitness_func,  # Custom fitness function
        sol_per_pop=20,  # Number of solutions per population
        num_genes=len(gene_space),  # Number of parameters to optimize
        gene_space=gene_space,  # Space of each gene (parameter)
        parent_selection_type="sss",  # Stochastic uniform selection
        keep_parents=2,  # Number of parents to keep in the next generation
        crossover_type="single_point",  # Crossover method
        mutation_type="random",  # Mutation method
        mutation_percent_genes=10,  # Percentage of genes to mutate
        on_generation=on_generation,  # Callback after each generation
        on_stop=on_stop  # Callback when GA stops
    )

    ga_instance.run()

    # After the run, you can get the best solution and its fitness value:
    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    print("Best solution:", solution)
    print("Fitness value of the best solution:", solution_fitness)
