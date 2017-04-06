import numpy as np
import scipy as sp
import pandas as pd
import ipdb

# Path
data_path = '../Data/'
file_train = data_path + 'train.csv'
file_test = data_path + 'test.csv'

def sign_log1p_abs(x):
    """Rescale data to smaller range and avoid the outliers skew away result.
    """
    return np.sign(x) * np.log1p(np.abs(x))

def DataFrame_tocsr(df, row, col, val=None, label2row=None, label2col=None,
                    return_tbl=False, min_count=1):
    """ Convert 2 or 3 colmns of a DataFrame to csr. Two columns are converted to row and column separaetely 
    and the optional third column is the value. The row and col labels are mapped to integers, used as indice. 
    col labels are sorted by frequency, such that the labels that appears the most frequently is at column 0.
    row lables are not sorted.

    Args:
         df: DataFrame, has 2 or 3 columns
         row: a column name to be used as row in output csr matrix
         col: a column name to be used as col in output csr matrix
         val: a column name to be used as values in output csr matrix
         label2row: function, dict, or Series to map row label to indice
         label2col: function, dict, or Series to map col label to indice
         return_tbl: return label2row and label2col if True
         min_count: minimum number of count in col to be included in output

    Returns:
         mat: csr matrix The columns are sorted by their frequency(decending).
         label2row: a map from a row label to a row number of mat
         label2column: a map from a column label to a column number of mat
    """
    #ipdb.set_trace()
    if label2row is None:
        row_labels = df[row].dropna().unique() # pd.Series.unique does not sort, all unique labels in the row
        label2row = pd.Series(range(row_labels.size), index=row_labels) #map label to a number
    if val is None:
        df = df[[row, col]].dropna()
        vals = pd.Series(np.ones(df.shape[0]))
    else:
        df = df[[row, col, val]].dropna()
        vals = df[val].values
    if label2col is None:
        col_label_cnt = df[col].value_counts() # count frequency of values in col
        if min_count > 1:
            col_label_cnt = col_label_cnt[col_label_cnt >= min_count]
        col_labels = col_label_cnt.index
        label2col = pd.Series(range(col_labels.size), index=col_labels)
    rows = df[row].map(label2row)
    cols = df[col].map(label2col)
    if cols.size == 0:
        return False
    mat = sp.sparse.coo_matrix((vals, (rows, cols)), shape=(label2row.size, label2col.size)).tocsr()
    if return_tbl:
        return mat, label2row, label2col
    else:
        return mat

def feature_extraction(training=None, test=None, useUpc=False):
    """Extract feature from file_train and file_test.
    
    Args:
         training: DataFrame, training data set
         test: DataFrame, test data set
         useUpc: use Upc as a feature if True
    Returns:
         X: csr_matrix, (n_visitnumber, n_feature)
         target: label of the training data set
         v_train: index of training data set in X (must be substracted by 1 before slicing X) 
         v_test: index of test data set in X (must be substracted by 1 before slicing X) 
    """
    
    if training is None and test is None:
        training = pd.read_csv(file_train)
        test = pd.read_csv(file_test)

    v_train = training.VisitNumber.unique()
    num_train = v_train.size
    v_test = test.VisitNumber.unique()
    grouped_train = training.groupby('VisitNumber')
    target = grouped_train.TripType.first().values      # TripType of each visit in training set

    data_all = training.append(test)
    data_all = data_all.sort_values('VisitNumber')
    data_all.ScanCount=data_all.ScanCount.astype(float)

    w2int = pd.Series(range(7), index=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
    Weekday = data_all.groupby('VisitNumber').Weekday.first()

    data_all['ScanCount_log1p'] = sign_log1p_abs(data_all.ScanCount)
    data_all.loc[data_all.DepartmentDescription=='MENSWEAR', 'DepartmentDescription'] = 'MENS WEAR'
    data_all.DepartmentDescription.fillna('-1', inplace=True)
    data_all.FinelineNumber.fillna(-1, inplace=True)
    data_all.Upc.fillna(-1, inplace=True)

    X_wday = sp.sparse.coo_matrix(pd.get_dummies(Weekday.map(w2int)).values)   # Convert categorical variable into dummy/indicator variables
    N = X_wday.shape[0]                   # number of visits in training and test
    X_SC_sum = data_all.groupby('VisitNumber').ScanCount.sum().values.reshape((N,1))
    X_SC_sum_sign = data_all.groupby('VisitNumber').ScanCount.apply(lambda x:1 if x.sum()>0 else 0).values.reshape((N, 1))
    X_dept = DataFrame_tocsr(data_all,
                             row='VisitNumber',
                             col='DepartmentDescription',
                             val='ScanCount')

    X_fine = DataFrame_tocsr(data_all,
                             row='VisitNumber',
                             col='FinelineNumber',
                             val='ScanCount_log1p')


    fine_dept = data_all[['FinelineNumber', 'DepartmentDescription']].drop_duplicates()
    fine_dept_cnt = fine_dept.FinelineNumber.value_counts()              # one FinelineNumber corresponds to multimple DepartmentDescription
    tmp = data_all.DepartmentDescription + '_' + data_all.FinelineNumber.astype(str)
    tmp[data_all.FinelineNumber.isin(fine_dept_cnt[fine_dept_cnt<2].index)] = np.nan # make Dept_Fine nan if FinelineNumber appeared less than 2 times
    data_all['Dept_Fine'] = tmp
    X_dept_fine = DataFrame_tocsr(data_all,
                                  row='VisitNumber',
                                  col='Dept_Fine',
                                  val='ScanCount_log1p')

    W_int = w2int.ix[Weekday]
    W_diff = W_int.diff().fillna(0)
    W_diff[W_diff!=0] = 1
    day = (W_diff.cumsum() + 1).values    # the first day is 1, the second day is 2 ...
    X_day = pd.get_dummies(day)
    # quasiHour_float = []
    # for i in range(1, 32):
    #     tmp = day[day == i]
    #     n = tmp.size
    #     quasiHour_float = np.append(quasiHour_float, np.arange(n) / float(n))
    # hour = (quasiHour_float * 24).astype(int)

    X = sp.sparse.hstack((X_day, X_SC_sum_sign, sign_log1p_abs(X_SC_sum),
                          X_dept, X_fine, X_dept_fine)).tocsr()
    if useUpc:
        X_upc = DataFrame_tocsr(data_all,
                                row='VisitNumber',
                                col='Upc',
                                val='ScanCount_log1p')
        X = sp.sparse.hstack((X, X_upc)).tocsr()
    return X, target, v_train, v_test
