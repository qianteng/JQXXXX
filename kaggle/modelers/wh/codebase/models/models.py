from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
import xgboost as xgb
from sklearn import preprocessing
import numpy as np
import pandas as pd


class RandomForestMultiClass(BaseEstimator):
    def __init__(self, n_estimators=50, n_jobs=4):
        self.clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs)

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X):
        preds = self.clf.predict_proba(X)
        classes = self.clf.classes_
        return preds, classes


class XGBoostMutliClass(BaseEstimator):
    def __init__(self, nthread, eta, max_depth, num_round, silent, booster='gbtree'):
        self.silent = silent
        self.nthread = nthread
        self.eta = eta
        self.silent = silent
        self.num_round=num_round
        self.num_classes = None
        self.model = None
        self.le = preprocessing.LabelEncoder()
        self.max_depth = max_depth
        self.booster = booster
        '''
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.seed = seed
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg
        '''

    def fit(self, X, y):
        y = self.le.fit_transform(y)
        self.num_classes = np.unique(y).shape[0]
        sf = xgb.DMatrix(X, y)
        params = {"objective": 'multi:softprob',
            "eta": self.eta,
            "silent": self.silent,
            "max_depth": self.max_depth,
            "num_class": self.num_classes,
            "booster": self.booster}
        '''
        "gamma": self.gamma,
         "min_child_weight": self.min_child_weight,
         "max_delta_step": self.max_delta_step,
         "subsample": self.subsample,

         "colsample_bytree": self.colsample_bytree,
         "seed": self.seed,
         "lambda": self.l2_reg,
         "alpha": self.l1_reg
        '''
        self.model = xgb.train(params, sf, self.num_round)
        return self

    def predict_proba(self, X):

        xg_test = xgb.DMatrix(X)
        y_probs = self.model.predict(xg_test).reshape(X.shape[0], self.num_classes)
        classes = self.le.inverse_transform([i for i in range(self.num_classes)])
        return y_probs, classes


class PyTorch(BaseEstimator):
    pass
