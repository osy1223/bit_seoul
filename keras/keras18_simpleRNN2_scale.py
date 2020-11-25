# 2. 실습 SimpleRNN 완성
# LSTM vs SimpleRNN  예측값 80, loss, 제일 좋은 파라미터 표 처럼 비교

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

x = x.reshape(13, 3, 1)
# print("x.shape : ", x.shape)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

model = Sequential()
model.add(SimpleRNN(80, input_shape=(3, 1)))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(30))
model.add(Dense(1))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=800, batch_size=8)

# x_input = np.array([50,60,70])
# print("x_input.shape : ", x_input.shape) # x_input.shape :  (3,)

x_input = x_input.reshape(1, 3, 1)
# print("x_input.shape : ", x_input.shape) #x_input.shape :  (1, 3, 1)

# 4. 예측
predict = model.predict(x_input)
print("predict :", predict)

'''
predict : [[73.661575]]
'''


loss = model.evaluate(x_input, np.array([80]), batch_size=7)
print("loss :", loss)

'''
loss : 40.175628662109375
'''