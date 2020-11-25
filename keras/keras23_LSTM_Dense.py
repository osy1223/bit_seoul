# Dense 모델로 

import numpy as np

# 1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
            [5,6,7], [6,7,8], [7,8,9], [8,9,10],
            [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_input = np.array([50,60,70])

print("x.shape : ", x.shape) # x.shape :  (13, 3)
print("y.shape : ", y.shape) # y.shape :  (13,)

#shape 맞추기
#LSTM을 사용하기 위해선 reshape가 필수불가결
#dense층에선 (13, 3)을 각 1열씩이라고 판단 가능하므로 reshape 필요 x

# x = x.reshape(13, 3, 1)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(Dense(30, activation='relu', input_dim=3)) #column 개수=3
model.add(Dense(70, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping #조기종료

early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')
model.fit(x, y, epochs=1000, batch_size=1, verbose=1, callbacks=[early_stopping])

# 4. 예측
x_input = x_input.reshape(1,3)
predict = model.predict(x_input)
print("predict :", predict)

loss = model.evaluate(x_input, np.array([80]), batch_size=1)
print("loss :", loss)
