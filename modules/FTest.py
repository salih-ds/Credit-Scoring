## F-TEST ##

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif

# Числовые
def f_num(data, cols):
    imp_num = pd.Series(f_classif(data[cols], data['default'])[0], index = cols)
    imp_num.sort_values(inplace = True)
    imp_num.plot(kind = 'barh', color='pink')

# Категориальные и бинарные
def f_cat(data, cols):
    imp_cat = pd.Series(mutual_info_classif(data[cols], data['default'],
                                     discrete_features =True), index = cols)
    imp_cat.sort_values(inplace = True)
    imp_cat.plot(kind = 'barh', color='pink')
