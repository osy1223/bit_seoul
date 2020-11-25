# Pipeline, make_pipeline

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

warnings.filterwarnings('ignore')

# 1. 데이터
iris = pd.read_csv('./data/csv/iris_ys.csv',
    header=0,
    index_col=0)

x = iris.iloc[:, :4] #(150,4)
y = iris.iloc[:, 4] ##(150, )

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=66
)

# svc용 파라미터

parameters = [
        {"svc__C": [1, 10, 100, 1000], "svc__kernel": ["linear"]}, #4가지
        {"svc__C": [1, 10, 100], "svc__kernel": ["rbf"], "svc__gamma":[0.001, 0.0001]}, # rbf 당 감마 4 * 2 = 8번
        {"svc__C": [1, 10, 100, 1000], "svc__kernel": ["sigmoid"], "svc__gamma":[0.001, 0.0001]} # sigmoid 당 감마 4 * 2 = 8번
            ]

# 2. 모델
pipe = make_pipeline(MinMaxScaler(), SVC()) 
# pipe = Pipeline([('scaler', MinMaxScaler()), ('malddong', SVC())])

# SVC 모델을 쓰는데, MinMaxScaler 스케일링을 쓰겠다. (모델+스케일링)

model = RandomizedSearchCV(pipe, parameters, cv=5)
# cv가 잘라주는 범위

# 3. 훈련
model.fit(x_train, y_train)
# cv=5를 적용시켜서 fit 하겠다

# 4. 평가, 예측
print('acc:', model.score(x_test, y_test))
print('최적의 모델 :', model.best_estimator_)
print('최적의 파라미터 :', model.best_params_)

'''
acc: 1.0
최적의 모델 : Pipeline(steps=[('minmaxscaler', MinMaxScaler()),
                ('svc', SVC(C=10, kernel='linear'))])
최적의 파라미터 : {'svc__kernel': 'linear', 'svc__C': 10}     
'''

