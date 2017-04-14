from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


def select_k(k, X_train, y_train, X_test):
    selector = SelectKBest(f_classif, k=k)
    selector.fit(X_train, y_train)
    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)
    return X_train, X_test