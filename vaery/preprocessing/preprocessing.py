import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import FunctionTransformer

class VAEPreprocessor(BaseEstimator):
    def __init__(self, other=[]):
        self.binary = None
        self.normal = None
        self.other = other
        self.columns = None
        self.vae_preprocessor = None
        self.result = None
        self.components = []
        
    def get_column_types(self, X, y=None):
        self.columns = X.columns.values
        self.binary = X.select_dtypes(include=['O']).columns.values
        normal = X.select_dtypes(include=['int', 'float']).columns.values
        if len(self.other) == 0:
            self.normal = normal
        else:
            self.normal = normal[~np.isin(normal, self.other)]

    def fit(self, X, y=None):
        self.get_column_types(X)
        binary_transformer = Pipeline(steps=[('encoder', OrdinalEncoder())])
        normal_transformer = Pipeline(steps=[('normal_scaler', MinMaxScaler())])
        log_transformer = FunctionTransformer(np.log1p)
        other_transformer = Pipeline(steps=[('other_log', log_transformer), ('other_scaler', MinMaxScaler())])
        self.vae_preprocessor = ColumnTransformer(transformers=
                                         [('binary', binary_transformer, self.binary), 
                                          ('normal', normal_transformer, self.normal), 
                                          ('other', other_transformer, self.other)],
                                         remainder='passthrough')
        self.vae_preprocessor.fit(X)
        
        return self
        
    def transform(self, X, y=None):
        result = pd.DataFrame(self.vae_preprocessor.transform(X), 
                              columns=np.append(np.append(self.binary, self.normal), self.other))
        result = result.fillna(0)
        result = result.replace([np.inf, -np.inf], 0)
        result = result
        
        return result
    
    def fit_transform(self, X, y=None):
        self.fit(X)

        return self.transform(X)
    
    def inverse_transform(self, X):
        X[self.binary] = np.abs(np.round(X[self.binary]))
        try:
            X_binary = pd.DataFrame(self.vae_preprocessor.transformers_[0][1].inverse_transform(X[self.binary]), 
                                    columns=self.binary)
            self.components.append(X_binary)
        except NotFittedError:
            pass
        
        try:
            X_normal = pd.DataFrame(self.vae_preprocessor.transformers_[1][1].inverse_transform(X[self.normal]), 
                                    columns=self.normal)
            self.components.append(X_normal)
        except NotFittedError:
            pass
                
        if self.other != []:
            X_other = self.vae_preprocessor.transformers_[2][1][1].inverse_transform(X[self.other])
            X_other = np.exp(X_other) - 1
            X_other = pd.DataFrame(X_other, columns=self.other)
            self.components.append(X_other)
            
        result = self.components[0]
        for i in range(len(self.components)-1):
            result = pd.merge(result, self.components[i+1], how='outer', left_index=True, right_index=True)
        #print(result)
        return result[self.columns]