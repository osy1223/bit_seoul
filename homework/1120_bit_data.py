import numpy as np
import pandas as pd
from pandas import DataFrame

# csv 파일 불러오기 (인코딩 주의)
bit = pd.read_csv('./data/csv/비트컴퓨터 1120.csv',
            header=0, index_col=0, sep=',', encoding='cp949')

print('bit:',bit)
# [1200 rows x 16 columns]

# 일부 데이터만 가져오기
bit = bit[['시가','고가','저가','종가']]
print(bit)
# [1200 rows x 4 columns]

# 일자 기준으로 오름차순으로 변경
bit = bit.sort_values(['일자'], ascending=['True'])
print(bit)

# 삼성이랑 데이터 개수 맞춰 가져오기
bit = bit.iloc[39:659]
print(bit)

#콤마 제거 후 문자를 정수로 변환
for i in range(len(bit.index)):
    for j in range(len(bit.iloc[i])):
        bit.iloc[i,j] = int(bit.iloc[i,j].replace(',',''))
print(bit)
# [1199 rows x 4 columns]

# numpy 파일로 변경 후 저장

bit = bit.to_numpy()
print(type(bit))

np.save('./data/bit.npy', arr=bit)
print(bit.shape)

# <class 'numpy.ndarray'>
# (1199, 4)
