# iris 다중 분류 모델 -> KNeighborsClassifier, RandomForestClassifier 사용

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# -Classifier : 분류모델, -Regressor : 회귀모델 
# 예외 : logistic regression은 분류모델
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
# x, y = load_iris(return_X_y=True)
datasets = load_iris()
print('feature_names:',datasets.feature_names)
# feature_names: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
print('target_names:',datasets.target_names)
# target_names: ['setosa' 'versicolor' 'virginica']

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, shuffle=True, train_size=0.8
)

scale = StandardScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)

# 2. 모델
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = KNeighborsRegressor()
model = RandomForestClassifier()
# model = RandomForestRegressor()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
score = model.score(x_test, y_test)
print('model.score:', score)

y_predict = model.predict(x_test)
print('y_predict:',y_predict)

#accuracy_score 분류모델 평가할 때
acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score:',acc_score)

#r2_score 회귀모델 평가할 때
# r2_score = r2_score(y_test, y_predict)
# print('r2_score:', r2_score)

print(y_test[:10], '의 예측 결과:','\n', y_predict[:10])

'''
model = RandomForestClassifier()
model.score: 0.9666666666666667
y_predict: [1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 1 2]
accuracy_score: 0.9666666666666667
[1 1 1 0 1 1 0 0 0 2] 의 예측 결과:
 [1 1 1 0 1 1 0 0 0 2]

model = KNeighborsClassifier()
model.score: 0.9
y_predict: [1 1 2 0 1 1 0 0 0 2 2 2 0 1 2 0 1 2 2 2 0 1 1 2 1 2 0 0 2 2]
accuracy_score: 0.9
[1 1 1 0 1 1 0 0 0 2] 의 예측 결과:
 [1 1 2 0 1 1 0 0 0 2]

model = LinearSVC()
model.score: 0.9333333333333333
y_predict: [1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 1 1 2]
accuracy_score: 0.9333333333333333
[1 1 1 0 1 1 0 0 0 2] 의 예측 결과:
 [1 1 1 0 1 1 0 0 0 2]

model = SVC()
model.score: 0.9333333333333333
y_predict: [1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 2 2 2 0 1 1 2 1 2 0 1 2 2]
accuracy_score: 0.9333333333333333
[1 1 1 0 1 1 0 0 0 2] 의 예측 결과:
 [1 1 1 0 1 1 0 0 0 2]

model = KNeighborsRegressor() -> 회귀 모델 
model.score: 0.9073825503355705
y_predict: [1.  1.2 1.6 0.  1.2 1.  0.  0.  0.  1.6 1.8 2.  0.  1.4 1.8 0.  1.  1.6
 2.  2.  0.  1.2 1.2 1.6 1.  2.  0.  0.4 1.8 2. ]
r2_score: 0.9073825503355705
[1 1 1 0 1 1 0 0 0 2] 의 예측 결과:
 [1.  1.2 1.6 0.  1.2 1.  0.  0.  0.  1.6]

model = RandomForestRegressor() -> 회귀 모델
model.score: 0.9505553691275168
y_predict: [1.   1.1  1.   0.   1.   1.   0.   0.   0.   1.25 2.   2.   0.   1.64
 1.99 0.   1.   1.28 2.   2.   0.   1.   1.   1.91 1.   2.   0.   0.
 1.56 2.  ]
r2_score: 0.9505553691275168
[1 1 1 0 1 1 0 0 0 2] 의 예측 결과:
 [1.   1.1  1.   0.   1.   1.   0.   0.   0.   1.25]
'''

