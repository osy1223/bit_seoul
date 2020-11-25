'''
winequality-white.csv
RF로 모델을 만들것
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

# 1. 데이터 가져오기
wine = pd.read_csv('./data/csv/winequality-white.csv',
        header=0, index_col=0, sep=';')

print(wine)
'''
               volatile acidity  citric acid  residual sugar  chlorides  free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  alcohol  quality
fixed acidity
7.0                        0.27         0.36            20.7      0.045                 45.0                 170.0  1.00100  3.00       0.45      8.8        6
6.3                        0.30         0.34             1.6      0.049                 14.0                 132.0  0.99400  3.30       0.49      9.5        6
8.1                        0.28         0.40             6.9      0.050                 30.0                  97.0  0.99510  3.26       0.44     10.1        6
7.2                        0.23         0.32             8.5      0.058                 47.0                 186.0  0.99560  3.19       0.40      9.9        6
7.2                        0.23         0.32             8.5      0.058                 47.0                 186.0  0.99560  3.19       0.40      9.9        6
...                         ...          ...             ...        ...                  ...                   ...      ...   ...        ...      ...      ...
6.2                        0.21         0.29             1.6      0.039                 24.0                  92.0  0.99114  3.27       0.50     11.2        6
6.6                        0.32         0.36             8.0      0.047                 57.0                 168.0  0.99490  3.15       0.46      9.6        5
6.5                        0.24         0.19             1.2      0.041                 30.0                 111.0  0.99254  2.99       0.46      9.4        6
5.5                        0.29         0.30             1.1      0.022                 20.0                 110.0  0.98869  3.34       0.38     12.8        7
6.0                        0.21         0.38             0.8      0.020                 22.0                  98.0  0.98941  3.26       0.32     11.8        6

[4898 rows x 11 columns]
'''
print(wine.shape) #(4898, 11)

# 2. 데이터 전처리
# x,y축 나누기
x = wine.iloc[:,:-1]
print('x:',x)

y = wine[['quality']]
print(y)


# train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7
)

print('x_train.shape:', x_train.shape)
print('x_test.shape:', x_test.shape)

# 3. 모델
model = RandomForestClassifier()

# 4. 훈련
model.fit(x_train, y_train)

# 5. 평가, 예측
score = model.score(x_test, y_test)
print('model.score :', score)

y_predict = model.predict(x_test)

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score:',acc_score)

'''
model = RandomForestClassifier()
model.score : 0.6578231292517007
accuracy_score: 0.6578231292517007
'''