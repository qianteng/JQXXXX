import pandas as pd
import xgboost
from sklearn import cross_validation
from sklearn.metrics import accuracy_score

data_path = r"C:\Users\Yijia\GITPROJ\JQXXXX\kaggle\modelers\yw\Data/pima-indians-diabetes.csv"

# load data
dataset = pd.read_csv(data_path)
# split data into X and y
X = dataset.values[:,:8]
Y = dataset.values[:,8]
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=test_size, random_state=seed)
# fit model no training data
model = xgboost.XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))