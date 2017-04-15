### Walmart Recruiting: Trip Type Classification

### Summary
24000 features by XGBoost. The current scores are: 0.63755 (private board), 0.64676 (public board), which ranks around 150. 

### Data Cleaning
* The NANs in UPC are filled with 0.
* The NANs in fineline number  are filled with -1
* The NANs in department are filled with 'UNKNOWN'

### Features
~60000 features are generated, 24000 of which are selected using ANOVA F-value

#### General Features
* Sum of scancount per visit
* number of returned items per visit
* number of bought items per visit
* min / max / mean / total number of times bought from each department per visit
* number of distinct department covered per visit
* number of distinct fineline numbers covered per visit
* number of distinct UPC covered per visit
* max number of items purchased under a single fineline number per visit
* the fineline number under which the most times are purchased per visit
* max number of items purchased under a single department per visit
* the department under which the most items are purchased per visit
#### Categorical Features
Categorical features like weekday, fineline number, department, upc, are one-hot encoded into dummy variables and compressed using sparse matrix representation (Scipy.sparse.csc_matrix())

* Weekday: 7 
* Fineline number: ~4000
* Department: 68
* UPC: select by quantile > 0.1 ~60000
* The department under which the most items are purchased per visit (From general features)

#### TF-IDF Features
TF-IDF features are extracted in two ways: (1) repeat the text in selected field by the absolute value of scancount (2) use the text in selected field as is.
* TF-IDF on department
* TF-IDF on fineline

However, it turns out that adding TF-IDF features downgrade the performance of models, therefore they are not used in the prediction. The code can be reused for the next contest

### Models
#### Random forest
The features are initially tested on Random forest, since there are very limited number of parameters to tune on RF and the training speed is fast.  It can handle sparse data too.

#### XGBoost
* Use csc sparse matrix as input, as xgboost has known errors in handling csr matrix
* Setting nthread can achieve mutli-processing instead of multi-threading, which is good.

### Parameter tuning
#### XGBoost
* nthread = 8
* eta = 0.1
* max_depth = 10
* num_round = 300
* **gamma = 0.925**

To Do:
* eta < 0.1
* min_child_weight

### Discussion
* Use of general features and Label encoding obtain the initial private score of 2.37. 
* Use of One-hot encoding on department and fineline number(quantile > 0.6) obtain a score of 0.82
* Adding more fineline number (quantile > 0.1) obtain a score of 0.7
* Adding UPC and doing feature selection (24000) then apply parameter tuning reaches a score of 0.637 
* **One-hot encoding of categorical variables, and use as many as them **
* **Parameter tuning is key**
