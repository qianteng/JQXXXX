from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
import pandas as pd

class RandomForest(BaseEstimator):
    def __init__(self, n_estimators=None, n_jobs=None):
        self.n_estimators = 50 if n_estimators is None else n_estimators
        self.n_jobs = 4 if n_jobs is None else n_jobs
        self.clf = RandomForestClassifier(n_estimators=self.n_estimators, n_jobs=self.n_jobs)

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X):
        preds = self.clf.predict_proba(X)
        classes = self.clf.classes_
        return preds, classes

    def tofile(self, preds, classes):
        pass
