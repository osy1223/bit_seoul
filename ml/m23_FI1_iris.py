# 기준 xgboost
# 1. FI 0 제거 -> 0 없으므로 하위 1개 제거
# 2. 피쳐 임포턴스를 재구성하여 디폴트와 성능 비교

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import  GradientBoostingClassifier, GradientBoostingRegressor
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. 데이터 
iris = load_iris()
print(iris.data)

# feature 
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'] 피쳐 4개
# label
#['setosa' 'versicolor' 'virginica']
 
x = iris.data
y = iris.target
print("init x.shape:", x.shape)


# 1.1 train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    iris.data, iris.target, train_size=0.8, random_state=42
)
print(x_train.shape) 
print(y_train.shape)


# 회귀모델
model = XGBClassifier(max_depth=4)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print('acc:',acc)
print(model.feature_importances_)


# 시각화
def plot_feature_importances(data_name, model):
    n_features = data_name.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), data_name.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)

plot_feature_importances(iris, model)
plt.show()

# print(model.feature_importances_) #[0.01145164 0.02792937 0.7579472  0.20267181]

#feautre importance(train 기준으로 슬라이스)
# sepal length

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



# 여명씨 도움받은 소스
# def get_drop_list(n):
#     a = model.feature_importances_.tolist()
#     b = a.copy()
#     b.sort()
#     b = b[:n] #앞에 몇개
#     index_list = []
#     for i in model.feature_importances_:
#         if i in b:
#             index_list.append(a.index(i))
#     return(index_list)

# drop_list = get_drop_list(3)
# print(drop_list)



x_train,x_test, y_train,y_test = train_test_split(
    x,y, random_state=44, shuffle=True, test_size=0.2)

model = XGBClassifier(max_depth=4)
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print("after erase low fi acc:", acc)


'''
위에건 피쳐임포턴스 구하려고 나누고 모델을 돌린거고
아래는, 피처숫자 줄인 x를 가지고, 다시 나눠서 다시 모델을 돌린거에얌 'ㅁ')/
'''
'''
acc: 1.0
after erase low fi acc: 0.9666666666666667
'''