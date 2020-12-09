# GRU

# 1. 데이터

import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
            [5,6,7], [6,7,8], [7,8,9], [8,9,10],
            [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_input = np.array([50,60,70])

print("x.shape : ", x.shape)

x = x.reshape(13,3,1)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU

model = Sequential()
model.add(GRU(10, activation='relu', input_shape=(3,1)))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=800, batch_size=8)

x_input = x_input.reshape(1, 3, 1)

# 4. 예측
predict = model.predict(x_input)
print("predict :", predict)


loss = model.evaluate(x_input, np.array([80]), batch_size=7)
print("loss :", loss)


