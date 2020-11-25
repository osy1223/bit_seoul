# csv파일을 np로 읽어오고 pd로 변환

# 실습
import numpy as np
import pandas as pd
from pandas import DataFrame

# iris_ys2.csv 파일을 np로 불러오기
datasets = np.loadtxt('./data/csv/iris_ys2.csv',
                delimiter=",", dtype=int)

print(datasets)
print(datasets.shape) #(150, 5)


# 불러온 데이터를 판다스(csv)로 저장하시오
# 파일명은 iris_ys2_pd.csv
dataframe = pd.DataFrame(datasets)
dataframe.to_csv("./data/csv/iris_ys2_pd.csv",
                header=False, index=False)

#데이터를 로드
data = pd.read_csv('./data/csv/iris_ys2_pd.csv')
print(data)
'''
     5  3  1  0  0.1
0    4  3  1  0    0
1    4  3  1  0    0
2    4  3  1  0    0
3    5  3  1  0    0
4    5  3  1  0    0
..  .. .. .. ..  ...
144  6  3  5  2    2
145  6  2  5  1    2
146  6  3  5  2    2
147  6  3  5  2    2
148  5  3  5  1    2
'''

# 모델 완성 (슬라이싱)
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense

