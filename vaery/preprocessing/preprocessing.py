import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

class VAEPreprocessor(BaseEstimator):
    def __init__(self):
        self.categorical = None
        self.numerical = None
        self.columns = None
        self.vae_preprocessor = None
        self.result = None
        
    def fit(self, X, y=None):
        self.categorical = X.select_dtypes(include=['O']).columns.values
        self.numerical = X.select_dtypes(include=['int', 'float']).columns.values
        self.columns = Xs.columns.values
        
        categorical_transformer = Pipeline(steps=[('encoder', OrdinalEncoder())])
        numerical_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])
        self.vae_preprocessor = ColumnTransformer(transformers=
                                         [('cat', categorical_transformer, self.categorical), 
                                          ('num', numerical_transformer, self.numerical)],
                                         remainder='passthrough')
        self.vae_preprocessor.fit(X)
        
        return self
        
    def transform(self, X, y=None):
        result = pd.DataFrame(self.vae_preprocessor.transform(X), columns=np.append(self.categorical, self.numerical))
        result = result.fillna(0)
        result = result.replace([np.inf, -np.inf], 0)
        self.result = result
        
        return self.result
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        
        return self.transform(X)
    
    def inverse_transform(self, X):
        X[self.categorical] = np.abs(np.round(X[self.categorical]))
        X_cat = pd.DataFrame(self.vae_preprocessor.transformers_[0][1].inverse_transform(X[self.categorical]), 
                             columns=self.categorical)
        X_num = pd.DataFrame(self.vae_preprocessor.transformers_[1][1].inverse_transform(X[self.numerical]), 
                             columns=self.numerical)
        X_num = np.abs(np.round(X_num))
        result = pd.merge(X_cat, X_num, how='outer', left_index=True, right_index=True)
        
        return result[self.columns]