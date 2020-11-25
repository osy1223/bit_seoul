# 유방암 데이터
# 모델 : RandomForestClassifier

import pandas as pd
import numpy as np
import warnings
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
cancer = load_breast_cancer()

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=66
)

parameters = [
    {'n_estimators' : [100,200]},
    {'max_depth' : [6,8,10,12]},
    {'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_split' : [2,3,5,10]},
    {'n_jobs':[-1]},
]
# 합집합 15가지 경우
# 'n_jobs':[-1]} : 모든 코어 쓰겠다 

parameters1 = [
    {'n_estimators' : [100,200],
    'max_depth' : [6,8,10,12],
    'min_samples_leaf' : [3,5,7,10],
    'min_samples_split' : [2,3,5,10],
    'n_jobs':[-1]},
]
# 교집합 모든 경우의 수 곱하기 128가지 경우
# 여러가지 경우의 수의 파라미터를 다 나오게 하려고 parmeter1을 사용!!!! (가장 좋은 파라미터 값 나오게 하려고)

# 2. 모델
kfold = KFold(n_splits=5, shuffle=True) #n_splits : 전체 데이터 중 몇개로 조각낼지
# model = SVC()
model = GridSearchCV(RandomForestClassifier(), parameters1, cv=kfold) #RandomForestClassifier 모델을 GridSearchCV로 쓰겠다

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
print('최적의 매개변수 :', model.best_estimator_)

y_predict = model.predict(x_test)
print('최종 정답률:', accuracy_score(y_test,y_predict))

'''
parameters 일 때, 
최적의 매개변수 : RandomForestClassifier()
최종 정답률: 0.9736842105263158

parameters1 일 때, 
최적의 매개변수 : RandomForestClassifier(max_depth=6, min_samples_leaf=3, min_samples_split=3,
                       n_jobs=-1)
최종 정답률: 0.9649122807017544
'''