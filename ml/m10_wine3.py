import pandas as pd
import matplotlib.pyplot as plt

wine = pd.read_csv('./data/csv/winequality-white.csv',
    sep=';', header=0)

count_data = wine.groupby('quality')['quality'].count()
#quality 그룹화 한 다음에 count 개수(숫자)를 세겠다
print(count_data)
'''
quality
3      20
4     163
5    1457
6    2198
7     880
8     175
9       5
Name: quality, dtype: int64
'''

count_data.plot()
plt.show()
