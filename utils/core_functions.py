import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERICAL_FEATURE = [
    'clin_size_long_diam_mm', 'age_approx', 'tbp_lv_A',
    'tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext', 'tbp_lv_C',
    'tbp_lv_Cext', 'tbp_lv_H', 'tbp_lv_Hext', 'tbp_lv_L',
    'tbp_lv_Lext', 'tbp_lv_areaMM2', 'tbp_lv_area_perim_ratio',
    'tbp_lv_color_std_mean', 'tbp_lv_deltaA', 'tbp_lv_deltaB',
    'tbp_lv_deltaL', 'tbp_lv_deltaLB', 'tbp_lv_deltaLBnorm',
    'tbp_lv_eccentricity', 'tbp_lv_minorAxisMM', 'tbp_lv_nevi_confidence',
    'tbp_lv_norm_border', 'tbp_lv_norm_color', 'tbp_lv_perimeterMM',
    'tbp_lv_radial_color_std_max', 'tbp_lv_stdL', 'tbp_lv_stdLExt',
    'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle']
#CATEGORICAL_FEATURES = ['sex','anatom_site_general', 'tbp_tile_type', 'tbp_lv_location', 'tbp_lv_location_simple', 'attribution']
CATEGORICAL_FEATURES = ['sex','anatom_site_general', 'tbp_tile_type', 'tbp_lv_location', 'tbp_lv_location_simple']
ID_COLUMNS = ['isic_id', 'patient_id']

def get_preprocessor(df_metadata):
    features = df_metadata.drop( columns = ['target','isic_id','patient_id'], axis=1)
    target = df_metadata['target']

    # Define the preprocessing pipeline
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Replace missing values with the mean
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Replace missing values with the mode
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, NUMERICAL_FEATURE),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )

    # Apply the preprocessing pipeline to the metadata
    features_preprocessed = preprocessor.fit_transform(features)

    # Combine preprocessed features with the target
    #metadata_preprocessed = pd.DataFrame(features_preprocessed)
    #metadata_preprocessed['target'] = target.values
    return preprocessor, features_preprocessed


class DataFrameTransformer(TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return pd.DataFrame(X, columns=self.feature_names)