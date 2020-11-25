#실습 LSTM 완성
#예측값 80
#함수형으로 코딩

# 1. 데이터

import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
            [5,6,7], [6,7,8], [7,8,9], [8,9,10],
            [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x_input = np.array([50,60,70])

print("x.shape : ", x.shape) #x.shape :  (13, 3)
print("y.shape : ", y.shape) #y.shape :  (13,)

x = x.reshape(13, 3, 1)
print("x.shape : ", x.shape) #x.shape :  (13, 3, 1)

x_input = np.array([50,60,70])
x_input = x_input.reshape(1,3,1)

#함수형 모델
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM

input1 = Input(shape=(3,1))
dense1 = LSTM(100, activation='relu')(input1)
dense2 = Dense(200, activation='relu')(dense1)
dense3 = Dense(50, activation='relu')(dense2)
dense4 = Dense(80, activation='relu')(dense3)
dense5 = Dense(10, activation='relu')(dense4)
output1 = Dense(1)(dense5)

model = Model(inputs=input1, outputs=output1)

model.summary()

# 3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam', metrics='mse')

model.fit(x, y, epochs=1000, batch_size=1)

# 4. 평가, 예측
y_predict = model.predict(x_input)
loss,acc = model.evaluate(x,y, batch_size=1)

print("predict :", y_predict)
print("loss : ", loss)
