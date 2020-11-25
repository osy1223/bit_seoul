# Pipeline, make_pipeline
# Randomforest

import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

# 1. 데이터
boston = load_boston()

x = boston.data #(506, 13)
y = boston.target #(506,)

print(x.shape)
print(y.shape)
print("y value category:",set(y)) #회귀모델

# 1.1 데이터 전처리
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.7, random_state=66
)

parameters = [
    {'malddong__n_estimators' : [100,200],
    'malddong__max_depth' : [6,8,10,12],
    'malddong__min_samples_leaf' : [3,5,7,9],
    'malddong__min_samples_split' : range(2,7),
    'malddong__n_jobs':[-1]},
]

# 2. 모델

pipe = Pipeline([('scaler', StandardScaler()), ('malddong', RandomForestRegressor())])

model = RandomizedSearchCV(pipe, parameters, cv=6, verbose=2) 

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
print('최적의 모델 :', model.best_estimator_)
print('최적의 파라미터 :', model.best_params_)

y_predict = model.predict(x_test)

# r2_score 회귀모델 평가할 때
r2_score = r2_score(y_test, y_predict)
print('r2_score:', r2_score)

'''
Pipeline 사용시,
최적의 모델 : Pipeline(steps=[('scaler', StandardScaler()),
                ('malddong',
                 RandomForestRegressor(max_depth=8, min_samples_leaf=3,
                                       min_samples_split=4, n_jobs=-1))])
최적의 파라미터 : {'malddong__n_jobs': -1, 'malddong__n_estimators': 100, 'malddong__min_samples_split': 4, 'malddong__min_samples_leaf': 3, 'malddong__max_depth': 8}
r2_score: 0.8741492233458787


make_pipeline 사용시,
'''