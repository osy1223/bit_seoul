import numpy as np

#1. 데이터 준비
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,3,4,5,6,7,8,9,10])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델 구성
model = Sequential()
model.add(Dense(30,input_dim=1))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(7))
model.add(Dense(1))

#3. 컴파일, 훈련 (컴퓨터가 알아들을 수 있도록)
# model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.compile(loss='mse', optimizer='adam', metrics=['mae', 'acc'])

model.fit(x, y, epochs=100)

#4. 평가, 예측
loss = model.evaluate(x,y)

print("loss : ", loss)
