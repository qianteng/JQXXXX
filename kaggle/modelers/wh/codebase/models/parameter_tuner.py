import models.models as models
from sklearn.metrics import log_loss


def xgb_tuner(X_train, y_train, X_test, y_test):
    for depth in [5, 10, 15, 20, 30]:
        for eta in [0.1, 0.15, 0.2, 0.3]:
            for num_round in [50, 100, 150]:
                xgb_clf = models.XGBoostMutliClass(nthread=8, eta=eta,
                                                   max_depth=depth, num_round=num_round, silent=0, booster='gbtree')
                xgb_clf.fit(X_train, y_train)
                yprob, preds = xgb_clf.predict_proba(X_test)
                logloss_score = log_loss(y_test, yprob)
                with open('xgb_grid_search.txt', 'a') as out_file:
                    out_file.write("eta=%.2f max_depth=%d num_round=%d log_loss=%f\n" % (eta, depth, num_round,
                                                                                         logloss_score))
