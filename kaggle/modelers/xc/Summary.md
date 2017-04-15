This is my first machine learning project. Well, the result is not good due to my lack of experience in python. But still, I have learned a lot from this practice project, such as installing python packages through pip, starting getting familiar with sklearn and XGBoost.

1. Data exploring:
Upc (universal product code): Usually refers to UPC-A which consists of 12 digits. The first 6 digits represents the company. The last digit is calculated by the other 11 digits. Unfortunately, not all the given Upc are in this UPC-A standard formula. The thing we can do is to fill leading zeros and obtain the company information of each product. 
FinelineNumber: It is Walmart self-defined product category label. It ranges from 0 to 9998.
Weekday: It can be considered as dummy variables.
ScantCount: Positive numbers mean purchasing, which negative numbers mean returning roducts. This could be used as one feature or sparated into two features. 
DepartmentDescription: There are 69 categories in the training data set and 68 categories in the test data set. The missing one is HEALTH AND BEAUTY AIDS. 

2. First try: 
The navie model is to use the Weekday and DepartmentDescription as dummy variables and perform logistic regression from sklearn package. It uses smaller number of features which saves training time, however, the accuracy is bad. The score from submission online is as high as 4.22834.

3. Second try:
Here, XGBoost package is used to do the training. In addition to the first try, ScantCount is also included as two features: purchase and return. Parameter tuning is done with hyperopt package. Thanks to XGBoost and the added two more features, the score from submission online is 2.78974. This is still far away from being good, but has been improved a lot from the first try.

4. Unfinished try:
According to what abhishekkrthakur mentioned in the kaggle discussion and github, he made use of the company information obtained from Upc and further split each feature into purchase and return. This created thousands of features and requires larger memory and longer training time. 
