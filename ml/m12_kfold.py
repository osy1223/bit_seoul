# KFold, cross_val_score

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


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

model = SVC()

scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('scores :', scores)
# scores : [1.         1.         0.91666667 0.95833333 0.95833333]

