# 4개의 모델을 완성 하시오
# KFold, cross_val_score

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# 1. 데이터
iris = pd.read_csv('./data/csv/iris_ys.csv',
    header=0, index_col=0)

x = iris.iloc[:, :4]
y = iris.iloc[:, -1]  #[:, 4]도 똑같습니다

print(x.shape, y.shape)
# (150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, shuffle=True, train_size=0.8
)

# 2. 모델

kfold = KFold(n_splits=5, shuffle=True) #n_splits : 전체 데이터 중 몇개로 조각낼지

model1 = SVC()
scores1 = cross_val_score(model1, x_train, y_train, cv=kfold)
print('SVC :', scores1)
# SVC : [0.95833333 1.         0.95833333 0.95833333 0.95833333]

model2 = LinearSVC()
scores2 = cross_val_score(model2, x_train, y_train, cv=kfold)
print('LinearSVC :', scores2)
# LinearSVC : [0.95833333 0.83333333 0.95833333 1.         0.95833333]

model3 = KNeighborsClassifier()
scores3 = cross_val_score(model3, x_train, y_train, cv=kfold)
print('KNeighborsClassifier :', scores3)
# KNeighborsClassifier : [0.95833333 0.91666667 0.95833333 0.95833333 1.        ]

model4 = RandomForestClassifier()
scores4 = cross_val_score(model4, x_train, y_train, cv=kfold)
print('RandomForestClassifier :', scores4)
# RandomForestClassifier : [0.91666667 0.91666667 0.95833333 1.         0.95833333]
