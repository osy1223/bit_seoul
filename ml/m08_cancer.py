# breast_cancer 분류모델 

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)


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
# model = RandomForestClassifier()
model = RandomForestRegressor()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
score = model.score(x_test, y_test)
print('model.score:', score)

y_predict = model.predict(x_test)
# print('y_predict:',y_predict)

#accuracy_score 분류모델 평가할 때
acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score:',acc_score)

#r2_score 회귀모델 평가할 때
r2_score = r2_score(y_test, y_predict)
print('r2_score:', r2_score)

# print(y_test[:10], '의 예측 결과:','\n', y_predict[:10])

'''
model = LinearSVC()
model.score: 0.9736842105263158
accuracy_score: 0.9736842105263158
r2_score: 0.885733377881724

model = SVC()
model.score: 0.9649122807017544
accuracy_score: 0.9649122807017544
r2_score: 0.8476445038422986

model = KNeighborsClassifier()
model.score: 0.956140350877193
accuracy_score: 0.956140350877193
r2_score: 0.8095556298028733

model = RandomForestClassifier()
model.score: 0.956140350877193
accuracy_score: 0.956140350877193
r2_score: 0.8095556298028733

model = KNeighborsRegressor()
model = RandomForestRegressor()
분류모델이니깐 ERROR
'''