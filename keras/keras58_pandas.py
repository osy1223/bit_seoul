# csv파일을 pd로 읽어오고 np로 변환

import numpy as np
import pandas as pd

datasets = pd.read_csv('./data/csv/iris_ys.csv', 
                        header=0, index_col=0, sep=',')
                                            # sep=','가 디폴트
print(datasets)
print(datasets.shape) #(150, 5)

# index_col = None, 0, 1 // header = None, 0, 1 일 때,
'''
header=0, index_col=None 일 때,
     Unnamed: 0  sepal_length  ...  petal_width  species
0             1           5.1  ...          0.2        0
1             2           4.9  ...          0.2        0
2             3           4.7  ...          0.2        0
3             4           4.6  ...          0.2        0
4             5           5.0  ...          0.2        0
..          ...           ...  ...          ...      ...
145         146           6.7  ...          2.3        2
146         147           6.3  ...          1.9        2
147         148           6.5  ...          2.0        2
148         149           6.2  ...          2.3        2
149         150           5.9  ...          1.8        2

[150 rows x 6 columns]
(150, 6)

header=0, index_col=0 일 때, 
     sepal_length  sepal_width  petal_length  petal_width  species
1             5.1          3.5           1.4          0.2        0
2             4.9          3.0           1.4          0.2        0
3             4.7          3.2           1.3          0.2        0
4             4.6          3.1           1.5          0.2        0
5             5.0          3.6           1.4          0.2        0
..            ...          ...           ...          ...      ...
146           6.7          3.0           5.2          2.3        2
147           6.3          2.5           5.0          1.9        2
148           6.5          3.0           5.2          2.0        2
149           6.2          3.4           5.4          2.3        2
150           5.9          3.0           5.1          1.8        2

[150 rows x 5 columns]
(150, 5)

header=0, index_col=1 일 때,
              Unnamed: 0  sepal_width  petal_length  petal_width  species   
sepal_length
5.1                    1          3.5           1.4          0.2        0   
4.9                    2          3.0           1.4          0.2        0   
4.7                    3          3.2           1.3          0.2        0   
4.6                    4          3.1           1.5          0.2        0   
5.0                    5          3.6           1.4          0.2        0   
...                  ...          ...           ...          ...      ...   
6.7                  146          3.0           5.2          2.3        2   
6.3                  147          2.5           5.0          1.9        2   
6.5                  148          3.0           5.2          2.0        2   
6.2                  149          3.4           5.4          2.3        2   
5.9                  150          3.0           5.1          1.8        2   

[150 rows x 5 columns]
(150, 5)

header=None, index_col=0 일 때,
                  1            2             3            4        5        
0
NaN    sepal_length  sepal_width  petal_length  petal_width  species        
1.0             5.1          3.5           1.4          0.2        0        
2.0             4.9            3           1.4          0.2        0        
3.0             4.7          3.2           1.3          0.2        0        
4.0             4.6          3.1           1.5          0.2        0        
...             ...          ...           ...          ...      ...        
146.0           6.7            3           5.2          2.3        2        
147.0           6.3          2.5             5          1.9        2        
148.0           6.5            3           5.2            2        2        
149.0           6.2          3.4           5.4          2.3        2        
150.0           5.9            3           5.1          1.8        2        

[151 rows x 5 columns]
(151, 5)

header=None, index_col=1 일 때, 
                  0            2             3            4        5        
1
sepal_length    NaN  sepal_width  petal_length  petal_width  species        
5.1             1.0          3.5           1.4          0.2        0        
4.9             2.0            3           1.4          0.2        0        
4.7             3.0          3.2           1.3          0.2        0        
4.6             4.0          3.1           1.5          0.2        0        
...             ...          ...           ...          ...      ...        
6.7           146.0            3           5.2          2.3        2        
6.3           147.0          2.5             5          1.9        2        
6.5           148.0            3           5.2            2        2        
6.2           149.0          3.4           5.4          2.3        2        
5.9           150.0            3           5.1          1.8        2        

[151 rows x 5 columns]
(151, 5)

header=None, index_col=None 일 때,
         0             1            2             3            4        5   
0      NaN  sepal_length  sepal_width  petal_length  petal_width  species   
1      1.0           5.1          3.5           1.4          0.2        0   
2      2.0           4.9            3           1.4          0.2        0   
3      3.0           4.7          3.2           1.3          0.2        0   
4      4.0           4.6          3.1           1.5          0.2        0   
..     ...           ...          ...           ...          ...      ...   
146  146.0           6.7            3           5.2          2.3        2   
147  147.0           6.3          2.5             5          1.9        2   
148  148.0           6.5            3           5.2            2        2   
149  149.0           6.2          3.4           5.4          2.3        2   
150  150.0           5.9            3           5.1          1.8        2   

[151 rows x 6 columns]
(151, 6)
'''

# header=None, index_col=None 이면, 없다는 생각으로 +1씩 됩니다 (데이터가 아니라고 판단)
# header=0, index_col=0 이면, header와 index를 정의해 준 겁니다

print(datasets.head())
'''
   sepal_length  sepal_width  petal_length  petal_width  species
1           5.1          3.5           1.4          0.2        0
2           4.9          3.0           1.4          0.2        0
3           4.7          3.2           1.3          0.2        0
4           4.6          3.1           1.5          0.2        0
5           5.0          3.6           1.4          0.2        0
'''
print(datasets.tail())
'''
     sepal_length  sepal_width  petal_length  petal_width  species
146           6.7          3.0           5.2          2.3        2
147           6.3          2.5           5.0          1.9        2
148           6.5          3.0           5.2          2.0        2
149           6.2          3.4           5.4          2.3        2
150           5.9          3.0           5.1          1.8        2
'''
print(type(datasets))

#datasets를 넘파이로 바꿀 것 2가지 방법 가능
aaa = datasets.to_numpy()
# aaa = datasets.values


print(type(aaa))
# <class 'numpy.ndarray'>
print(aaa.shape)

np.save('./data/iris_ys_pd.npy', arr=aaa)

