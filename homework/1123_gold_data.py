import numpy as np
import pandas as pd
from pandas import DataFrame

# csv 파일 불러오기 (인코딩 주의)
gold = pd.read_csv('./data/csv/금현물.csv',
            header=0, index_col=0, sep=',', encoding='cp949')

print('gold:',gold)
print(gold.columns.tolist())



# 일부 데이터만 가져오기
gold = gold[['고가','저가','종가','거래량','거래대금(백만)','시가']]
print(gold)


# 일자 기준으로 오름차순으로 변경
gold = gold.sort_values(['일자'], ascending=['True'])
print(gold)


# row 맞춰주기
gold = gold.iloc[39:659]
print(gold)
# [620 rows x 3 columns]

#콤마 제거 후 문자를 정수로 변환
for i in range(len(gold.index)):
    for j in range(len(gold.iloc[i])):
        gold.iloc[i,j] = int(gold.iloc[i,j].replace(',',''))
print(gold)

# numpy 파일로 변경 후 저장
gold = gold.to_numpy()
print(type(gold))
# <class 'numpy.ndarray'>

np.save('./data/gold.npy', arr=gold)
print(gold)
print(gold.shape) # (620, 6)