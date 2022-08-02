import pandas as pd
import numpy as np
import tensorflow as tf
tf.random.set_seed(1234)

from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import interp1d

class VAE(BaseEstimator, TransformerMixin):       
    def __init__(self, *, preprocessor=None, batch_size=1024, 
                 epochs=1000, n_epochs_without_progress=50, 
                 model=base_model(latent_dim=1), callbacks=None, verbose=1):
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_epochs_without_progress = n_epochs_without_progress
        self.model = model
        self.callbacks = callbacks
        self.verbose = verbose
        self.loss_history_ = []
        self.callbacks = callbacks
        
        self.callbacks_ = None
        self.encoder_ = None
        self.decoder_ = None
        self.samples_ = None
        self.decoded_samples_ = None
        
    def _validate_parameters(self):
        self.callbacks_ = [] if self.callbacks is None else list(map(copy, self.callbacks))

        #for callback in self.callbacks_:
        #    if isinstance(callback, ModelCheckpoint):
        #        callback.register_ivis_model(self)

        
    def _fit(self, X, Y=None, shuffle_mode=False):
        self._validate_parameters()
        if self.verbose > 0:
            print('Training neural network')

        hist = self.model.fit(
            X, X, ###
            epochs=self.epochs,
            callbacks=self.callbacks_ + [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       patience=self.n_epochs_without_progress, 
                                                       restore_best_weights=True)],
            validation_split=0.2, ###
            shuffle=shuffle_mode,
            steps_per_epoch=int(np.ceil(X.shape[0] / self.batch_size)),
            verbose=self.verbose)
        self.loss_history_ += hist.history['loss']
    
    def fit(self, X, y=None):
        #check_is_fitted(self.estimator)
        #self.reference_X = estimator.transform(X, y)
        if self.preprocessor is not None:
            self.preprocessor.fit(X)
            Xt = self.preprocessor.transform(X)
            
        else:
            Xt = X.copy()
            
        self._fit(Xt) 
        self.encoder_ = self.model.layers[1]
        self.decoder_ = self.model.layers[2]
        
    def fit_transform(self, X, y=None, shuffle_mode=False):
        self.fit(X, shuffle_mode)
        return self.transform(X)

    def transform(self, X):
        if self.encoder_ is None:
            raise NotFittedError("Model was not fitted yet. Call `fit` before calling `transform`.")
        
        if self.preprocessor is not None:
            Xt = self.preprocessor.transform(X)
            result = pd.DataFrame(self.model.predict(Xt), columns=self.preprocessor.columns)
            result = self.preprocessor.inverse_transform(result)
            
        else:
            Xt = X.copy()
            result = self.model.predict(Xt)
            
        return result
    
    def sample(self, X, N=1000):
        if self.encoder_ is None:
            raise NotFittedError("Model was not fitted yet. Call `fit` before calling `sample`.")

        encoder_pred = self.encoder_.predict(self.preprocessor.transform(X))
        ecdf = ECDF(encoder_pred[:, 0])
        x_min = encoder_pred.min()
        x_max = encoder_pred.max()
        
        x = np.linspace(x_min, x_max, 100, endpoint=True)
        
        edf_samples = [ecdf(i) for i in x]
        inverted_edf = interp1d(edf_samples, x)
        #print(np.max(edf_samples))
        
        self.samples_ = pd.DataFrame(self.decoder_(inverted_edf(np.random.uniform(np.min(edf_samples), np.max(edf_samples), N))).numpy(), 
                                    columns=self.preprocessor.columns)
        self.decoded_samples_ = self.preprocessor.inverse_transform(self.samples_)
        
        return self