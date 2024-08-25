import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.ensemble import EasyEnsembleClassifier, BalancedRandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, cross_val_score, StratifiedKFold
from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from utils.core_functions import NUMERICAL_FEATURE, CATEGORICAL_FEATURES, DataFrameTransformer

seed = 42

#CATEGORICAL_FEATURES = ['sex', 'anatom_site_general', 'tbp_tile_type', 'tbp_lv_location', 'tbp_lv_location_simple',
#                        'attribution']
#CATEGORICAL_FEATURES = ['sex']

xgb_params = {
    'enable_categorical': True,
    'tree_method': 'hist',
    'random_state': seed,
    'learning_rate': 0.08501257473292347,
    'lambda': 8.879624125465703,
    'alpha': 0.6779926606782505,
    'max_depth': 6,
    'subsample': 0.6012681388711075,
    'colsample_bytree': 0.8437772277074493,
    'colsample_bylevel': 0.5476090898823716,
    'colsample_bynode': 0.9928601203635129,
    'scale_pos_weight': 3.29440313334688,
}

cb_params = {
    'loss_function': 'Logloss',
    'iterations': 250,
    'verbose': False,
    'random_state': 40,
    'max_depth': 7,
    'learning_rate': 0.06936242010150652,
    'scale_pos_weight': 2.6149345838209532,
    'l2_leaf_reg': 6.216113851699493,
    'subsample': 0.6249261779711819,
    'min_data_in_leaf': 24,
    'cat_features': CATEGORICAL_FEATURES
}


def custom_metric(estimator, X, y_true):
    y_hat = estimator.predict_proba(X)[:, 1]
    min_tpr = 0.80
    max_fpr = abs(1 - min_tpr)
    v_gt = abs(y_true - 1)
    v_pred = np.array([1.0 - x for x in y_hat])
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr ** 2 + (max_fpr - 0.5 * max_fpr ** 2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    return partial_auc


df_train = pd.read_csv("_raw_data/train-metadata.csv")
df_train = df_train.drop(
    columns=['isic_id', 'patient_id', 'lesion_id', 'iddx_1', 'iddx_2', 'iddx_3', 'iddx_4', 'iddx_5', 'iddx_full',
             'mel_mitotic_index', 'mel_thick_mm', 'tbp_lv_dnn_lesion_confidence'])
X, y = df_train.drop(columns=['target']), df_train.target
X = X[NUMERICAL_FEATURE + CATEGORICAL_FEATURES]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))])

categorical_transformer_with_one_hot = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, NUMERICAL_FEATURE),
        ('cat', categorical_transformer, CATEGORICAL_FEATURES)])

preprocessor_with_one_hot = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, NUMERICAL_FEATURE),
        ('cat', categorical_transformer_with_one_hot, CATEGORICAL_FEATURES)])

ee_pipeline = ImbPipeline(steps=[
    ('ee_preprocessor', preprocessor_with_one_hot),
    ('ee_oversample', RandomOverSampler(random_state=seed, sampling_strategy=0.003)),
    ('ee_undersample', RandomUnderSampler(random_state=seed, sampling_strategy=0.01)),
    ('ee_classifier', EasyEnsembleClassifier(random_state=seed, n_jobs=10, n_estimators=100))], verbose=True)

cb_pipeline = ImbPipeline([
    ('cb_preprocessor', preprocessor),
    ('to_dataframe', DataFrameTransformer(feature_names=X.columns)),
    ('cb_oversample', RandomOverSampler(sampling_strategy=0.003, random_state=seed)),
    ('cb_undersample', RandomUnderSampler(sampling_strategy=0.01, random_state=seed)),
    ('cb_classifier', cb.CatBoostClassifier(**cb_params)),
], verbose=True)

xgb_pipeline = ImbPipeline([
    ('cb_preprocessor', preprocessor_with_one_hot),
    ('xgb_oversample', RandomOverSampler(sampling_strategy= 0.003 , random_state=seed)),
    ('xgb_undersample', RandomUnderSampler(sampling_strategy=0.01, random_state=seed)),
    ('xgb_classifier', xgb.XGBClassifier(**xgb_params)),
])

brf_pipeline = ImbPipeline([
    ('brf_preprocessor', preprocessor_with_one_hot),
    ('brf_oversample', RandomOverSampler(sampling_strategy=0.003, random_state=seed)),
    ('brf_undersample', RandomUnderSampler(sampling_strategy=0.01, random_state=seed)),
    ('brf_classifier', BalancedRandomForestClassifier(max_depth=100)),
], verbose=True)

estimator = VotingClassifier([
    #('brf', brf_pipeline),
    ('cb', cb_pipeline),
    #('ee', ee_pipeline),
    ('xgb', xgb_pipeline)
], voting='soft')

cv = StratifiedKFold(5, shuffle=True, random_state=seed)

val_score = cross_val_score(
    estimator=estimator,
    X=X, y=y,
    cv=cv,
    #groups=groups,
    scoring=custom_metric, verbose=1, n_jobs=8
)

print(np.mean(val_score))
