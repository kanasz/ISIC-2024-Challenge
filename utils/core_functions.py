import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import polars as pl

err = 1e-5

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
CATEGORICAL_FEATURES = ['sex', 'anatom_site_general', 'tbp_tile_type', 'tbp_lv_location', 'tbp_lv_location_simple']
ID_COLUMNS = ['isic_id', 'patient_id']
NEW_FEATURES = ["lesion_size_ratio", "lesion_shape_index", "hue_contrast", "luminance_contrast",
                "lesion_color_difference", "border_complexity", "color_uniformity",
                "position_distance_3d", "color_uniformity", "position_distance_3d", "perimeter_to_area_ratio",
                "area_to_perimeter_ratio", "lesion_visibility_score",
                #"combined_anatomical_site",
                "symmetry_border_consistency", "consistency_symmetry_border",
                "color_consistency", "consistency_color", "size_age_interaction",
                "hue_color_std_interaction", "lesion_severity_index", "shape_complexity_index", "color_contrast_index",
                "log_lesion_area", "normalized_lesion_size",
                "mean_hue_difference", "std_dev_contrast", "color_shape_composite_index", "lesion_orientation_3d",
                "overall_color_difference"]


def get_preprocessor(df_metadata):
    features = df_metadata.drop(columns=['target', 'isic_id', 'patient_id'], axis=1)
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
            ('num', numerical_transformer, NUMERICAL_FEATURE + NEW_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )

    # Apply the preprocessing pipeline to the metadata
    features_preprocessed = preprocessor.fit_transform(features)

    # Combine preprocessed features with the target
    #metadata_preprocessed = pd.DataFrame(features_preprocessed)
    #metadata_preprocessed['target'] = target.values
    return preprocessor, features_preprocessed


def get_data(metadata_path, lesion_confidence_threshold):
    df_metadata = pd.read_csv(metadata_path, index_col=False)
    df_metadata["lesion_size_ratio"] = df_metadata['tbp_lv_minorAxisMM'] / df_metadata['clin_size_long_diam_mm']
    df_metadata["lesion_shape_index"] = df_metadata['tbp_lv_areaMM2'] / (df_metadata['tbp_lv_perimeterMM'] ** 2)
    df_metadata["hue_contrast"] = (df_metadata['tbp_lv_H'] - df_metadata['tbp_lv_Hext']).abs()
    df_metadata["luminance_contrast"] = (df_metadata['tbp_lv_L'] - df_metadata['tbp_lv_Lext']).abs()
    df_metadata["lesion_color_difference"] = np.sqrt(
        df_metadata['tbp_lv_deltaA'] ** 2 + df_metadata['tbp_lv_deltaB'] ** 2 + df_metadata['tbp_lv_deltaL'] ** 2)
    df_metadata["border_complexity"] = df_metadata['tbp_lv_norm_border'] + df_metadata['tbp_lv_symm_2axis']
    df_metadata["color_uniformity"] = df_metadata['tbp_lv_color_std_mean'] / (
                df_metadata['tbp_lv_radial_color_std_max'] + err)

    df_metadata["position_distance_3d"] = np.sqrt(
        df_metadata['tbp_lv_x'] ** 2 + df_metadata['tbp_lv_y'] ** 2 + df_metadata['tbp_lv_z'] ** 2)
    df_metadata["perimeter_to_area_ratio"] = df_metadata['tbp_lv_perimeterMM'] / df_metadata['tbp_lv_areaMM2']
    df_metadata["area_to_perimeter_ratio"] = df_metadata['tbp_lv_areaMM2'] / df_metadata['tbp_lv_perimeterMM']
    df_metadata["lesion_visibility_score"] = df_metadata['tbp_lv_deltaLBnorm'] + df_metadata['tbp_lv_norm_color']
    #df_metadata["combined_anatomical_site"] = df_metadata['anatom_site_general'] + '_' + df_metadata['tbp_lv_location']
    df_metadata["symmetry_border_consistency"] = df_metadata['tbp_lv_symm_2axis'] * df_metadata['tbp_lv_norm_border']
    df_metadata["consistency_symmetry_border"] = df_metadata['tbp_lv_symm_2axis'] * df_metadata[
        'tbp_lv_norm_border'] / (df_metadata['tbp_lv_symm_2axis'] + df_metadata['tbp_lv_norm_border'])

    df_metadata["color_consistency"] = df_metadata['tbp_lv_stdL'] / df_metadata['tbp_lv_Lext']
    df_metadata["consistency_color"] = df_metadata['tbp_lv_stdL'] * df_metadata['tbp_lv_Lext'] / (
                df_metadata['tbp_lv_stdL'] + df_metadata['tbp_lv_Lext'])
    df_metadata["size_age_interaction"] = df_metadata['clin_size_long_diam_mm'] * df_metadata['age_approx']
    df_metadata["hue_color_std_interaction"] = df_metadata['tbp_lv_H'] * df_metadata['tbp_lv_color_std_mean']
    df_metadata["lesion_severity_index"] = (df_metadata['tbp_lv_norm_border'] + df_metadata['tbp_lv_norm_color'] +
                                            df_metadata['tbp_lv_eccentricity']) / 3
    df_metadata["shape_complexity_index"] = df_metadata['border_complexity'] + df_metadata['lesion_shape_index']
    df_metadata["color_contrast_index"] = df_metadata['tbp_lv_deltaA'] + df_metadata['tbp_lv_deltaB'] + df_metadata[
        'tbp_lv_deltaL'] + df_metadata['tbp_lv_deltaLBnorm']

    df_metadata["log_lesion_area"] = np.log(df_metadata['tbp_lv_areaMM2'] + 1)
    df_metadata["normalized_lesion_size"] = df_metadata['clin_size_long_diam_mm'] / df_metadata['age_approx']
    df_metadata["mean_hue_difference"] = (df_metadata['tbp_lv_H'] + df_metadata['tbp_lv_Hext']) / 2
    df_metadata["std_dev_contrast"] = np.sqrt(
        (df_metadata['tbp_lv_deltaA'] ** 2 + df_metadata['tbp_lv_deltaB'] ** 2 + df_metadata['tbp_lv_deltaL'] ** 2) / 3)
    df_metadata["color_shape_composite_index"] = (df_metadata['tbp_lv_color_std_mean'] + df_metadata[
        'tbp_lv_area_perim_ratio'] + df_metadata[
                                                      'tbp_lv_symm_2axis']) / 3
    df_metadata["lesion_orientation_3d"] = np.arctan2(df_metadata['tbp_lv_y'], df_metadata['tbp_lv_x'])
    df_metadata["overall_color_difference"] = (df_metadata['tbp_lv_deltaA'] + df_metadata['tbp_lv_deltaB'] +
                                               df_metadata['tbp_lv_deltaL']) / 3

    df_metadata = df_metadata[
        df_metadata['tbp_lv_dnn_lesion_confidence'] > lesion_confidence_threshold]
    df_metadata = df_metadata.reset_index()

    return df_metadata


'''
def read_data(path):
    return (
        pl.read_csv(path)
        .with_columns(
            pl.col('age_approx').cast(pl.String).replace('NA', np.nan).cast(pl.Float64),
        )
        .with_columns(
            pl.col(pl.Float64).fill_nan(pl.col(pl.Float64).median()), # You may want to impute test data with train
        )
        .with_columns(
            lesion_size_ratio              = pl.col('tbp_lv_minorAxisMM') / pl.col('clin_size_long_diam_mm'),
            lesion_shape_index             = pl.col('tbp_lv_areaMM2') / (pl.col('tbp_lv_perimeterMM') ** 2),
            hue_contrast                   = (pl.col('tbp_lv_H') - pl.col('tbp_lv_Hext')).abs(),
            luminance_contrast             = (pl.col('tbp_lv_L') - pl.col('tbp_lv_Lext')).abs(),
            lesion_color_difference        = (pl.col('tbp_lv_deltaA') ** 2 + pl.col('tbp_lv_deltaB') ** 2 + pl.col('tbp_lv_deltaL') ** 2).sqrt(),
            border_complexity              = pl.col('tbp_lv_norm_border') + pl.col('tbp_lv_symm_2axis'),
            color_uniformity               = pl.col('tbp_lv_color_std_mean') / (pl.col('tbp_lv_radial_color_std_max') + err),
        )
        .with_columns(
            position_distance_3d           = (pl.col('tbp_lv_x') ** 2 + pl.col('tbp_lv_y') ** 2 + pl.col('tbp_lv_z') ** 2).sqrt(),
            perimeter_to_area_ratio        = pl.col('tbp_lv_perimeterMM') / pl.col('tbp_lv_areaMM2'),
            area_to_perimeter_ratio        = pl.col('tbp_lv_areaMM2') / pl.col('tbp_lv_perimeterMM'),
            lesion_visibility_score        = pl.col('tbp_lv_deltaLBnorm') + pl.col('tbp_lv_norm_color'),
            combined_anatomical_site       = pl.col('anatom_site_general') + '_' + pl.col('tbp_lv_location'),
            symmetry_border_consistency    = pl.col('tbp_lv_symm_2axis') * pl.col('tbp_lv_norm_border'),
            consistency_symmetry_border    = pl.col('tbp_lv_symm_2axis') * pl.col('tbp_lv_norm_border') / (pl.col('tbp_lv_symm_2axis') + pl.col('tbp_lv_norm_border')),
        )
        .with_columns(
            color_consistency              = pl.col('tbp_lv_stdL') / pl.col('tbp_lv_Lext'),
            consistency_color              = pl.col('tbp_lv_stdL') * pl.col('tbp_lv_Lext') / (pl.col('tbp_lv_stdL') + pl.col('tbp_lv_Lext')),
            size_age_interaction           = pl.col('clin_size_long_diam_mm') * pl.col('age_approx'),
            hue_color_std_interaction      = pl.col('tbp_lv_H') * pl.col('tbp_lv_color_std_mean'),
            lesion_severity_index          = (pl.col('tbp_lv_norm_border') + pl.col('tbp_lv_norm_color') + pl.col('tbp_lv_eccentricity')) / 3,
            shape_complexity_index         = pl.col('border_complexity') + pl.col('lesion_shape_index'),
            color_contrast_index           = pl.col('tbp_lv_deltaA') + pl.col('tbp_lv_deltaB') + pl.col('tbp_lv_deltaL') + pl.col('tbp_lv_deltaLBnorm'),
        )
        .with_columns(
            log_lesion_area                = (pl.col('tbp_lv_areaMM2') + 1).log(),
            normalized_lesion_size         = pl.col('clin_size_long_diam_mm') / pl.col('age_approx'),
            mean_hue_difference            = (pl.col('tbp_lv_H') + pl.col('tbp_lv_Hext')) / 2,
            std_dev_contrast               = ((pl.col('tbp_lv_deltaA') ** 2 + pl.col('tbp_lv_deltaB') ** 2 + pl.col('tbp_lv_deltaL') ** 2) / 3).sqrt(),
            color_shape_composite_index    = (pl.col('tbp_lv_color_std_mean') + pl.col('tbp_lv_area_perim_ratio') + pl.col('tbp_lv_symm_2axis')) / 3,
            lesion_orientation_3d          = pl.arctan2(pl.col('tbp_lv_y'), pl.col('tbp_lv_x')),
            overall_color_difference       = (pl.col('tbp_lv_deltaA') + pl.col('tbp_lv_deltaB') + pl.col('tbp_lv_deltaL')) / 3,
        )
        .with_columns(
            symmetry_perimeter_interaction = pl.col('tbp_lv_symm_2axis') * pl.col('tbp_lv_perimeterMM'),
            comprehensive_lesion_index     = (pl.col('tbp_lv_area_perim_ratio') + pl.col('tbp_lv_eccentricity') + pl.col('tbp_lv_norm_color') + pl.col('tbp_lv_symm_2axis')) / 4,
            color_variance_ratio           = pl.col('tbp_lv_color_std_mean') / pl.col('tbp_lv_stdLExt'),
            border_color_interaction       = pl.col('tbp_lv_norm_border') * pl.col('tbp_lv_norm_color'),
            border_color_interaction_2     = pl.col('tbp_lv_norm_border') * pl.col('tbp_lv_norm_color') / (pl.col('tbp_lv_norm_border') + pl.col('tbp_lv_norm_color')),
            size_color_contrast_ratio      = pl.col('clin_size_long_diam_mm') / pl.col('tbp_lv_deltaLBnorm'),
            age_normalized_nevi_confidence = pl.col('tbp_lv_nevi_confidence') / pl.col('age_approx'),
            age_normalized_nevi_confidence_2 = (pl.col('clin_size_long_diam_mm')**2 + pl.col('age_approx')**2).sqrt(),
            color_asymmetry_index          = pl.col('tbp_lv_radial_color_std_max') * pl.col('tbp_lv_symm_2axis'),
        )
        .with_columns(
            volume_approximation_3d        = pl.col('tbp_lv_areaMM2') * (pl.col('tbp_lv_x')**2 + pl.col('tbp_lv_y')**2 + pl.col('tbp_lv_z')**2).sqrt(),
            color_range                    = (pl.col('tbp_lv_L') - pl.col('tbp_lv_Lext')).abs() + (pl.col('tbp_lv_A') - pl.col('tbp_lv_Aext')).abs() + (pl.col('tbp_lv_B') - pl.col('tbp_lv_Bext')).abs(),
            shape_color_consistency        = pl.col('tbp_lv_eccentricity') * pl.col('tbp_lv_color_std_mean'),
            border_length_ratio            = pl.col('tbp_lv_perimeterMM') / (2 * np.pi * (pl.col('tbp_lv_areaMM2') / np.pi).sqrt()),
            age_size_symmetry_index        = pl.col('age_approx') * pl.col('clin_size_long_diam_mm') * pl.col('tbp_lv_symm_2axis'),
            index_age_size_symmetry        = pl.col('age_approx') * pl.col('tbp_lv_areaMM2') * pl.col('tbp_lv_symm_2axis'),
        )
        .with_columns(
            ((pl.col(col) - pl.col(col).mean().over('patient_id')) / (pl.col(col).std().over('patient_id') + err)).alias(f'{col}_patient_norm') for col in (num_cols + new_num_cols)
        )
        .with_columns(
            count_per_patient = pl.col('isic_id').count().over('patient_id'),
        )
        #.with_columns(
        #    pl.col(cat_cols).cast(pl.Categorical),
        #)
        .to_pandas()
        #.set_index(id_col)
    )
'''


class DataFrameTransformer(TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return pd.DataFrame(X, columns=self.feature_names)
