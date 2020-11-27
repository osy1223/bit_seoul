import pandas as pd
import numpy as np
from pandas import DataFrame as df



#################################################################행복지수
happy = pd.read_csv('./project3/happiness.csv', 
                        header=0, index_col=0, sep=',')
# happy.info()
'''
Index: 147 entries, Afghanistan to Zimbabwe
Data columns (total 20 columns):
'''


# 결측치 바로 이전의 값으로 채우기
happy = happy.fillna(method='pad')
happy.info()
print(happy.shape) #(147, 20)


