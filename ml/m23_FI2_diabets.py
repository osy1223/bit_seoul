# 기준 xgboost
# 1. FI 0 제거 또는 하위 30% 제거
# 2. 디폴트와 성능 비교


from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import  GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print("init x.shape:",x.shape)

# train_test_split
x_train,x_test, y_train,y_test = train_test_split(
    x,y, random_state=44, shuffle=True, test_size=0.2)

# 모델 
model = XGBRegressor(max_depth=4)
model.fit(x_train, y_train)

# 평가 
acc = model.score(x_test, y_test)
print("acc:", acc)
print(model.feature_importances_)


# 피쳐 임포턴스 자르기 함수 
def earseLowFI_index(fi_arr, low_value, input_arr):
    input_arr = input_arr.T
    temp = []
    for i in range(fi_arr.shape[0]):
        if fi_arr[i] >= low_value:
            temp.append(input_arr[i,:])
    temp = np.array(temp)
    temp = temp.T
    return temp

print("before x.shape:",x.shape)
x = earseLowFI_index(model.feature_importances_, 0.09, x)
print("after x.shape:",x.shape)


# 다시 자르기 
x_train,x_test, y_train,y_test = train_test_split(
    x,y, random_state=44, shuffle=True, test_size=0.2)

# 다시 모델 
model = XGBRegressor(max_depth=4)
model.fit(x_train, y_train)

# 평가 
acc = model.score(x_test, y_test)
print("after erase low fi acc:", acc)

'''
acc: 0.2436866612630486
after erase low fi acc: 0.12242171873116647
'''