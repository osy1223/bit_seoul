# reshape, input_length, input_dim

# 1. 데이터
import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]]) 
y = np.array([4,5,6,7]) 

print("x.shape : ", x.shape) # x.shape:(4, 3)
print("y.shape : ", y.shape) # y.shape:(4,)

x = x.reshape(x.shape[0], x.shape[1], 1)

print("x.shape : ", x.shape)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(10, input_length=3, input_dim=1))
model.add(Dense(40))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(90))
model.add(Dense(200))
model.add(Dense(1))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='mse')
model.fit(x,y, epochs=100, batch_size=6)

x_input = np.array([5,6,7]) #(3,) -> (1, 3, 1)
#데이터 reshape
x_input = x_input.reshape(1, 3, 1)

# 예측
y_predict = model.predict(x_input)
loss, acc = model.evaluate(x, y, batch_size=1)

print("예측값: ", y_predict)
print("loss: ", loss, "\n", "acc: ", acc)


'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 10)                480       
_________________________________________________________________
dense (Dense)                (None, 40)                440
_________________________________________________________________
dense_1 (Dense)              (None, 80)                3280
_________________________________________________________________
dense_2 (Dense)              (None, 60)                4860
_________________________________________________________________
dense_3 (Dense)              (None, 90)                5490
_________________________________________________________________
dense_4 (Dense)              (None, 200)               18200
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 201
=================================================================
Total params: 32,951
Trainable params: 32,951
Non-trainable params: 0
_________________________________________________________________
예측값:  [[7.2476]]
loss:  0.028337836265563965
acc:  0.028337836265563965
'''