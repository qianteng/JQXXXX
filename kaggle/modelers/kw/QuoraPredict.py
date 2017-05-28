import numpy as np, pandas as pd, scipy as sp, sklearn as skl # , matplotlib.pyplot as plt 
import re
import xgboost as xgb
from sklearn.metrics import log_loss
from collections import Counter
from nltk.corpus import stopwords
stops = set(stopwords.words("english"))

split2Words = lambda x:re.findall('[\w]+', x)

def preproc(df):
    df.loc[:,'question1'] = df['question1'].fillna('')
    df.loc[:,'question2'] = df['question2'].fillna('')
    return df 


train_orig = pd.read_csv('data_quora/train.csv')
test_orig = pd.read_csv('data_quora/test.csv') 
df= preproc(train_orig.copy()) 


def makeFeature1(r): # shared common words
    wds1, wds2 = split2Words(r['question1'].lower()), split2Words(r['question2'].lower())
    c1, c2 = Counter([x for x in wds1 if x not in stops]).keys(), Counter([x for x in wds2 if x not in stops]).keys()
    if len(c1)== 0 or len(c2) == 0 :
        return 0 
    else:
        return (len([x for x in c1 if x in c2]) + len([x for x in c2 if x in c1])) / float(len(c1) + len(c2))

def get_weight(count, eps=1000, min_count=2):
    if count < min_count:
        return 0.
    else:
        return 1. / (count + eps)

def makeFeature3(r, weights):
    #print r 
    wds1, wds2 = split2Words(r['question1'].lower()), split2Words(r['question2'].lower())
    c1, c2 = Counter([x for x in wds1 if x not in stops]).keys(), Counter([x for x in wds2 if x not in stops]).keys()
    if len(c1)== 0 or len(c2) == 0 :
        return 0 
    else:
        return (np.sum([weights.get(x,0.) for x in c1 if x in c2]) + np.sum([weights.get(x,.0) for x in c2 if x in c1])
               ) / (np.sum([weights.get(x,.0) for x in c1]) +np.sum([weights.get(x,.0) for x in c2]) )
       
def makeFeature6(r): 
    wds1, wds2 = split2Words(r['question1']), split2Words(r['question2'])
    c1, c2 = [x for i, x in enumerate( wds1) if x.lower() not in stops and x.lower()<>x and i <> 0] , [
        x for i, x in enumerate(wds2) if x.lower() not in stops and x.lower()<>x and i <> 0] 
    return len([x for x in c1 if x in c2])

def makeFeature7(r):
    wds1, wds2 = split2Words(r['question1']), split2Words(r['question2'])
    c1, c2 = [x for x in wds1 if x.isdigit()], [x for x in wds2 if x.isdigit()]
    return len([x for x in c1 if x in c2])

def makeFeatures(df) :
    #if 0:
    x_train = pd.DataFrame()
    x_train.loc[:,'f1'] = df.apply(makeFeature1, axis=1, raw=True)  # shared word % 
    eps = 5000 
    words = split2Words((" ".join(df.question1.values).lower() + ' '.join(df.question2.values)).lower())
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}
    x_train.loc[:,'f3'] = df.apply(lambda x:makeFeature3(x, weights), axis=1, raw=True).fillna(0)  # tfidf shared % 
    #x_train.loc[:,'f2'] = df.apply(makeFeature2, axis=1)
    if 'is_duplicate' in df.columns:
        q1freq = Counter(df.qid1)
        q2freq = Counter(df.qid2)
        x_train.loc[:,'f4'] = np.int32(df.qid1.map(q1freq) / 10.) # freq of q1
        x_train.loc[:,'f5'] = np.int32(df.qid2.map(q2freq) / 10.) # freq of q2 
    else: 
        q1freq = Counter(df.question1)
        q2freq = Counter(df.question2)
        x_train.loc[:,'f4'] = np.int32(df.question1.map(q1freq) / 10.) # freq of q1
        x_train.loc[:,'f5'] = np.int32(df.question2.map(q2freq) / 10.) # freq of q2         
    x_train.loc[:,'f6'] = df.apply(makeFeature6, axis=1, raw=True).fillna(0)  # cap words 
    x_train.loc[:,'f7'] = df.apply(makeFeature7, axis=1, raw=True).fillna(0)  # cap words
    return x_train
    
#return x_train 
x_train = makeFeatures(df) 
y_train = df.is_duplicate.values
from sklearn.cross_validation import train_test_split 
x_tr, x_va, y_tr, y_va = train_test_split(x_train[['f1','f3','f4','f5', 'f6', 'f7']].values
    , y_train, test_size=0.3, random_state=3462)

params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 6
d_tr = xgb.DMatrix(x_tr, label=y_tr, weight = 1 - 0.5 * y_tr)
d_va = xgb.DMatrix(x_va, label=y_va, weight = 1 - 0.5 * y_va )
watchlist = [(d_tr, 'train'), (d_va, 'valid')]
bst = xgb.train(params, d_tr, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)


dfTest = preproc(test_orig.copy())
x_test = makeFeatures(dfTest)[['f1','f3','f4','f5','f6','f7']].values
d_te = xgb.DMatrix(x_test)
pred_te = bst.predict(d_te)

sub = pd.DataFrame({'test_id': dfTest['test_id'], 'is_duplicate': pred_te})[['test_id','is_duplicate']]
sub.to_csv('sub.csv', index=False)









