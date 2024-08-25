import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tabular_tuning.tabular_tuning_constants import SEED, NUMERICAL_FEATURE, CATEGORICAL_FEATURES
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_data():
    df_train = pd.read_csv("_raw_data/train-metadata.csv")
    groups = df_train['patient_id']
    df_train = df_train.drop(
        columns=['isic_id', 'patient_id', 'lesion_id', 'iddx_1', 'iddx_2', 'iddx_3', 'iddx_4', 'iddx_5', 'iddx_full',
                 'mel_mitotic_index', 'mel_thick_mm', 'tbp_lv_dnn_lesion_confidence'])
    X, y = df_train.drop(columns=['target']), df_train.target
    X = X[NUMERICAL_FEATURE + CATEGORICAL_FEATURES]
    return X, y, groups


def custom_metric(estimator, X, y_true):
    y_hat = estimator.predict_proba(X)[:, 1]
    min_tpr = 0.80
    max_fpr = abs(1 - min_tpr)
    v_gt = abs(y_true - 1)
    v_pred = np.array([1.0 - x for x in y_hat])
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr ** 2 + (max_fpr - 0.5 * max_fpr ** 2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    return partial_auc


def get_preprocessor(use_one_hot=False):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))])

    if use_one_hot:
        categorical_transformer.steps.append(('ohe', OneHotEncoder(handle_unknown='ignore')))

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERICAL_FEATURE),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)])
    return preprocessor


def run_experiment(pipeline, X, y, groups):
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)

    val_score = cross_val_score(
        groups=groups,
        estimator=pipeline,
        X=X, y=y,
        cv=cv,
        scoring=custom_metric, verbose=0, n_jobs=8
    )

    return np.mean(val_score)