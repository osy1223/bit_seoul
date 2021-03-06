# diabets 선형회귀모델(Linear Regression)
# 당뇨병 환자의 1년 후 병의 진전된 정도를 예측하는 모델

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


# 1. 데이터
x, y = load_diabetes(return_X_y=True)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, shuffle=True, train_size=0.8
)

scale = StandardScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)

# 2. 모델
# model = LinearSVC()
model = SVC()
# model = KNeighborsClassifier()
# model = KNeighborsRegressor()
# model = RandomForestClassifier()
# model = RandomForestRegressor()

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
model = RandomForestClassifier()
model.score: 0.02247191011235955
accuracy_score: 0.02247191011235955

model = KNeighborsClassifier()
model.score: 0.0
accuracy_score: 0.0

model = KNeighborsRegressor()
model.score: 0.38626977834604637
r2_score: 0.38626977834604637

model = RandomForestRegressor()
model.score: 0.36084507744865
r2_score: 0.36084507744865

model = LinearSVC()
model.score: 0.011235955056179775
accuracy_score: 0.011235955056179775
r2_score: -0.15630588545158997

model = SVC()
model.score: 0.0
accuracy_score: 0.0
r2_score: -0.0833765078750015
'''