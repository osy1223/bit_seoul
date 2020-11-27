# 기준 xgboost
# x_train + x_test 하고 feture importance
# 실행 3번
# 디폴트
# 0인 것 제거 또는 하위 30% 제거
# 3개의 성능 비교
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRFRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# 1. 데이터 
datasets = load_boston()
x = datasets.data
y = datasets.target
print("init x.shape:",x.shape)

# 1.1 데이터 전처리 (train_test_split)
x_train,x_test, y_train,y_test = train_test_split(
    x,y, random_state=44, shuffle=True, test_size=0.2)

# 2 모델 (XGBRFRegressor)
model = XGBRFRegressor(max_depth=4)
model.fit(x_train, y_train)

# 4. 평가
acc = model.score(x_test, y_test)
print("acc:", acc)
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

print("before x.shape:",x.shape)
x = earseLowFI_index(model.feature_importances_, 0.1, x)
print("after x.shape:",x.shape)


# 다시 train_test 분리
x_train,x_test, y_train,y_test = train_test_split(
    x,y, random_state=44, shuffle=True, test_size=0.2)

# 모델
model = XGBRFRegressor(max_depth=4)
model.fit(x_train, y_train)

# 재평가
acc = model.score(x_test, y_test)
print("after erase low fi acc:", acc)


'''
acc: 0.8479204761653626
after erase low fi acc: 0.8059016044333062
'''