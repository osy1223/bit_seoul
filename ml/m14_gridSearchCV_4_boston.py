# 실습
# 보스톤 (회귀)
# 파일을 gridSearchCV 3, 4, 5 만들 것
# RandomForest로 만들것 

import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
boston = load_boston()

x = boston.data #(506, 13)
y = boston.target #(506,)

print(x.shape) 
print(y.shape) 

# 1-1 데이터 전처리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=66
)

parameters = [
    {'n_estimators' : [100,200],
    'max_depth' : [6,8,10,12],
    'min_samples_leaf' : [3,5,7,10],
    'min_samples_split' : [2,3,5,10],
    'n_jobs':[-1]},
]

# 2. 모델
kfold = KFold(n_splits=5, shuffle=True) #n_splits : 전체 데이터 중 몇개로 조각낼지
# model = SVC()
model = GridSearchCV(RandomForestRegressor(), parameters, cv=kfold) 
#RandomForestRegressor 모델을 GridSearchCV로 쓰겠다

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
print('최적의 매개변수 :', model.best_estimator_)

y_predict = model.predict(x_test)

# r2_score 회귀모델 평가할 때
r2_score = r2_score(y_test, y_predict)
print('r2_score:', r2_score)


'''
최적의 매개변수 : RandomForestRegressor(max_depth=10, min_samples_leaf=3, n_jobs=-1)
r2_score: 0.9188137724954061
'''