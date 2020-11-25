# 케라스로 와인 모델을 완성하시오

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

# 1. 데이터 가져오기
wine = pd.read_csv('./data/csv/winequality-white.csv',
        header=0, index_col=0, sep=';')

print(wine)
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

print('x_train.shape:', x_train.shape) #x_train.shape: (3428, 10)
print('x_test.shape:', x_test.shape) #x_test.shape: (1470, 10)


# scaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print('x_train.shape:', x_train.shape) #x_train.shape: (3428, 10)
print('x_test.shape:', x_test.shape) 


# categorical (OneHotEncoding)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print('y_train.shape:',y_train.shape) #y_train.shape: (3428, 10)

# reshape
x_train = x_train.reshape(3428, 10, 1)
x_test = x_test.reshape(1470, 10, 1)
print('x_train.reshape:', x_train.shape) #x_train.reshape: (3428, 10, 1)
print('x_test.reshape:', x_test.shape) 

# 3. 모델
model = Sequential()
model.add(LSTM(1, input_shape=(10,1)))
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(500))
model.add(Dense(200))
model.add(Dense(10, activation='softmax'))

model.summary()

# 4. 훈련
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['acc']
)

model.fit(x_train, y_train,
    epochs=100,
    batch_size=5,
    verbose=1,
    validation_split=0.5   
)

# 5. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ",loss)
print("acc : ",acc)

y_predict = model.predict(x_test) # 평가 데이터 다시 넣어 예측값 만들기

# 원핫인코딩 후 원복
y_test = np.argmax(y_test, axis=1)
y_predict = np.array(y_predict, axis=1)
print("y_test:\n", y_test)
print("y_predict:\n", y_predict)

'''
loss :  1.1612937450408936
acc :  0.5
'''