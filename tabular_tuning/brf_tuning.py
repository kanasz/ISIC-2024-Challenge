import warnings
import optuna.visualization as vis
from optuna.samplers import NSGAIISampler, TPESampler, QMCSampler, BaseSampler

warnings.simplefilter(action='ignore', category=FutureWarning)

import optuna
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from tabular_tuning.tabular_tuning_constants import SEED
from tabular_tuning.tabular_tuning_functions import get_preprocessor, get_data, run_experiment

X, y, groups = get_data()


def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 5, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 30)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 30)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    sampling_strategy = trial.suggest_categorical('sampling_strategy', ['majority', 'not minority', 'not majority', 'all'])
    oversampling_strategy = trial.suggest_float('oversampling_strategy', 0.003, 0.009)
    pipeline = ImbPipeline([
        ('brf_preprocessor', get_preprocessor(True)),
        #('brf_oversample', RandomOverSampler(sampling_strategy=oversampling_strategy, random_state=SEED)),
        ('brf_smote', SMOTE(random_state=SEED, k_neighbors=10, n_jobs=5, sampling_strategy=oversampling_strategy)),
        #('brf_undersample', RandomUnderSampler(sampling_strategy=0.01, random_state=SEED)),

        ('brf_classifier', BalancedRandomForestClassifier(criterion=criterion,
                                                          random_state=SEED,
                                                          max_depth=max_depth,
                                                          n_estimators=n_estimators,
                                                          min_samples_split=min_samples_split,
                                                          max_features=max_features,
                                                          min_samples_leaf=min_samples_leaf,
                                                          sampling_strategy=sampling_strategy)),
    ], verbose=False)

    result = run_experiment(pipeline, X, y, groups)
    return result
sampler = TPESampler(seed=SEED)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=50)

print("Best trial:")
trial = study.best_trial

print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Plot optimization history
vis.plot_optimization_history(study)

# Plot parameter importance
vis.plot_param_importances(study)

# Plot the Pareto front for multi-objective optimization
vis.plot_pareto_front(study)