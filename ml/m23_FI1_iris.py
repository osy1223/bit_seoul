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

# data_min = np.min(iris.data)
# print(data_min) #0.1

# data_max = np.max(iris.data)
# print(data_max) #7.9

# feature 
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'] 피쳐 4개
# label
#['setosa' 'versicolor' 'virginica']

print(iris.data.shape) #(150, 4)
print(iris.target.shape) #(150,)  



'''
# 1.1 train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    iris.data, iris.target, train_size=0.8, random_state=42
)
print(x_train.shape) #(120, 4)
print(y_train.shape) #(120,)


model = XGBClassifier(max_depth=4)
# max_depth=4 : 4번 잘랐다

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

print(acc)
print(model.feature_importances_)


# 시각화
def plot_feature_importances(model):
    n_features = iris.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), iris.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)

plot_feature_importances(model)
plt.show()

# print(model.feature_importances_) #[0.01145164 0.02792937 0.7579472  0.20267181]

#feautre importance(train 기준으로 슬라이스)
# sepal length
'''

