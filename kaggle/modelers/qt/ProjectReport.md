#[Walmart Recruiting: Trip Type Classification Project](https://www.kaggle.com/c/walmart-recruiting-trip-type-classification)
## [Feature Engineering](https://github.com/tks0123456789/kaggle-Walmart_Trip_Type)
1. There are 6 features in the training and test data set.
  * VisitNumber is just and id used to group trades of a single customer
  * Weekday is the weekday of the trip. By combining the test and train data set, it's obvious that the whole data set is from a month. A new feature of date is constructed.
  * Upc is the universal product code. The standard Upc should be 12 digits. I expanded many of the Upcs in the data set to 12 digits. Then the numbering system feature and company code feature are extracted.
  * ScanCount is the number of the given item that was purchased. By summing ScanCount, the total buy or return feature is constructed.
  * DepartmentDescription, a high-level description of the item's department.
  * FinelineNumber, is determined by the sales pattern of a product. It makes sense to create a new feature as a combination of DepartmentDescription and FinelineNumber.
2. It's memory efficient to store catagorical data set into sparse matrix.
3. Scale data to small range with function log1p help avoid the result be skewed by outliers a lot.

## [Parameter Tuning](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/?cm_mc_uid=41931017955214914245309&cm_mc_sid_50200000=1491424530)
1. Choose a relatively high learning rate. Generally a learning rate of 0.1 works but somewhere between 0.05 to 0.3 should work for different problems. Determine the optimum number of trees for this learning rate. XGBoost has a very useful function called as “cv” which performs cross-validation at each boosting iteration and thus returns the optimum number of trees required.
2. Tune tree-specific parameters ( max_depth, min_child_weight, gamma, subsample, colsample_bytree) for decided learning rate and number of trees. Note that we can choose different parameters to define a tree and I’ll take up an example here.
3. Tune regularization parameters (lambda, alpha) for xgboost which can help reduce model complexity and enhance performance.
4. Lower the learning rate and decide the optimal parameters.
5. Use xgb.cv to tune num_round. Use GridSearchV or RandomizedSearchCV to tune the other parameters.

## Conclusion
The program is developed on top of a top solution by tks. His ranking is 15, with score 0.52625. He used an ensemble of Neural Network and xgboost. I only worked on xgboost. Taking his xgboost part directly, the score is 0.61620. By adding some feature engineering on Upc, I achieved score of 0.59764. Through some parameter tuning, I achieved score of 0.59553.
## Appendix
* Install multiprocessing xgboost on mac.  
  brew install gcc --without-multilib    git clone --recursive https://github.com/dmlc/xgboost    cd xgboost    cp make/config.mk ./config.mk    In config.mk, change    export CC = gcc    export CC = g++    to    export CC = gcc-6     export CC = g++-6    uncomment the two changed lines
* [Set up virtual enviroment with conda](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/). WalmartEnv is the virtualenv I use for this project.
* [Using AWS](http://docs.aws.amazon.com/gettingstarted/latest/awsgsg-intro/gsg-aws-tutorials.html)
* Set up working environment on AWS.  
yum install gcc gcc-c++ autoconf automake  pip install xgboost=0.6a2  
* [One bug in xgboost](https://github.com/dmlc/xgboost/issues/1238). When using CSR matrix, xgboost can produce wrong result. As there might be empty columns and xgboost.DMatrix can't handle that well. One way to work around this is to use CSC matrix. But there is still some problem with using GridSearchCV.fit(X, y).