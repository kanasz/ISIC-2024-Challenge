import time
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class MRMRFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=10, mode='MIQ'):
        self.n_features = n_features
        self.mode = mode
        self.selected_features_ = None

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        df.columns = df.columns.astype(str)
        #df['target'] = y
        from mrmr import mrmr_classif

        self.selected_features_ = mrmr_classif(X=df, y=y, K=self.n_features, show_progress=False)
        #self.selected_features_ = pymrmr.mRMR(df, self.mode, self.n_features)
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        df.columns = df.columns.astype(str)
        return df[self.selected_features_]