# 실습
# 유방암 (분류모델)
# 파일을 randomSearch 2, 3, 4, 5 만들 것
# RandomForest로 만들것 

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
cancer = load_breast_cancer()

x = cancer.data
y = cancer.target

print(x.shape) #(569, 30)
print(y.shape) #(569,)
print("y value category:",set(y)) #y value category: {0, 1}


# 1-1 데이터 전처리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66
)

parameters = [
    {'n_estimators' : [100,200],
    'max_depth' : [6,8,10,12],
    'min_samples_leaf' : [3,5,7,9],
    'min_samples_split' : range(2,7),
    'n_jobs':[-1]},
]

# 2. 모델
kfold = KFold(n_splits=8, shuffle=True) #n_splits : 전체 데이터 중 몇개로 조각낼지
# model = SVC()
model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=2) 
#RandomForestClassifier 모델을 RandomizedSearchCV로 쓰겠다

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
print('최적의 매개변수 :', model.best_estimator_)

y_predict = model.predict(x_test)
print('최종 정답률:', accuracy_score(y_test,y_predict))

'''
최적의 매개변수 : RandomForestClassifier(max_depth=6, min_samples_leaf=9, min_samples_split=4,
                       n_jobs=-1)
최종 정답률: 0.9532163742690059
'''
