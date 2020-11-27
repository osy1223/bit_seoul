# 기준 xgboost
# 1. FI 0 제거 또는 하위 30% 제거
# 2. 디폴트와 성능 비교
# wine csv파일 사용


from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import  GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRFRegressor
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# 1. 데이터
wine = pd.read_csv('./data/csv/winequality-white.csv',
    header=0,
    index_col=0,
    sep=';')

y = wine['quality']
x = wine.drop('quality', axis=1).to_numpy()
print(x.shape) #(4898, 10)
print(y.shape) #(4898,)

# 1.1 데이터 전처리 (train_test_split)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=44, shuffle=True, test_size=0.2
)

# 2. 모델
model = XGBClassifier(max_depth=4)
model.fit(x_train, y_train)

# 4. 평가, 예측
y_predict = model.predict(x_test)
print('최종 정답률 :', r2_score(y_test, y_predict))

acc = model.score(x_test, y_test)
print('acc:', acc)
print(model.feature_importances_)

# 피쳐 임포턴스 자르는 함수
def earseLowFI_index(fi_arr, low_value, input_arr):
    input_arr = input_arr.T
    temp = []
    for i in range(fi_arr.shape[0]):
        if fi_arr[i] >= low_value:
            temp.append(input_arr[i,:])
    temp = np.array(temp)
    temp = temp.T
    return temp

print('before x.shape:', x.shape)
x = earseLowFI_index(model.feature_importances_, 0.08, x)
print('after x.shape:', x.shape)

# 다시 train, test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=44, shuffle=True, test_size=0.2
)

model = XGBClassifier(max_depth=4)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
print('after erase lof fi 최종 정답률 :',  r2_score(y_test, y_predict))

'''
최종 정답률 : 0.40178165030236046
after erase lof fi 최종 정답률 : 0.29904415111515714
'''