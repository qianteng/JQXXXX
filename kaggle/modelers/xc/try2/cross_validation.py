import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss
from hyperopt import fmin, tpe, hp, STATUS_OK 

def objective1(params):
	print "Training model1 with parameters: "
	print params
	watchlist1 = [(dtrain1, 'train'), (dtestCV1, 'eval')]
	model = xgb.train(params=params, 
		dtrain=dtrain1, 
		num_boost_round=100,  
		early_stopping_rounds=10, 
		evals=watchlist1)
	score = log_loss(dtestCV1.get_label(), model.predict(dtestCV1))
	print "\tScore {0}\n\n".format(score)
	return {'loss': score, 'status': STATUS_OK}

def objective2(params):
	print "Training model2 with parameters: "
	print params
	watchlist2 = [(dtrain2, 'train'), (dtestCV2, 'eval')]
	model = xgb.train(params=params, 
		dtrain=dtrain2, 
		num_boost_round=100,  
		early_stopping_rounds=10, 
		evals=watchlist2)
	score = log_loss(dtestCV2.get_label(), model.predict(dtestCV2))
	print "\tScore {0}\n\n".format(score)
	return {'loss': score, 'status': STATUS_OK}

if __name__ == "__main__":
        """load data"""
	dtrain1 = xgb.DMatrix('dtrain1.buffer')
	dtestCV1 = xgb.DMatrix('dtestCV1.buffer')
	dtrain2 = xgb.DMatrix('dtrain2.buffer')
	dtestCV2 = xgb.DMatrix('dtestCV2.buffer')

        """Set the hyperparameter space"""
	space = {'eta' : hp.quniform('eta', 0.025, 0.5, 0.025),
		'max_depth' : hp.choice('max_depth', np.arange(1, 16, dtype=int)),
		'min_child_weight' : hp.quniform('min_child_weight', 1, 6, 1),
		'subsample' : hp.quniform('subsample', 0.5, 1, 0.05),
		'gamma' : hp.quniform('gamma', 0.5, 1, 0.05),
		'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05),
		'num_class' : 38,
		'eval_metric': 'mlogloss',
		'objective': 'multi:softprob'}

	"""Evaluate the loss and find the optima parameters"""  
	best1 = fmin(objective1, space=space, algo=tpe.suggest, max_evals=100)
	print "Optimal parameters for dtrain1 are: ", best1

	best2 = fmin(objective2, space=space, algo=tpe.suggest, max_evals=100)
	print "Optimal parameters for dtrain2 are: ", best2

