#!/usr/bin/env Python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import glob
import ipdb

train = pd.read_csv('Data/train.csv', dtype = {'TripType': str, 'VisitNumber': str, 'Weekday': str, 'Upc': str,
                                               'ScanCount': np.int32, 'DepartmentDescription': str, 'FinelineNumber': str})

# Upc and FinelineNumber were typically only missing for rows with "NULL" description, or with "PHARMACY RX"
# This is true
# train[train['Upc'].isnull()].to_csv('test.csv')

# Records with 999 TripType appear to mostly be related to trips specifically for returning or exchanging items (sum of ScanCount <= 0)
# This is not quite true
# other = train[train['TripType']=='999'].groupby('VisitNumber')['ScanCount'].sum()
# entrys, counts = np.unique(other, return_counts = True)
# df = pd.DataFrame({'entrys': entrys, 'counts': counts})
# plt.plot(entrys, counts, '-+')
# plt.show()
