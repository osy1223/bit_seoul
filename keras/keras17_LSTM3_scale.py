# 2. 실습 LSTM 완성
# 원하는 예측값 80 

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
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(30, activation='relu', input_shape=(3, 1))) 
model.add(Dense(70))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='mse')
model.fit(x, y, epochs=250, batch_size=1)

# 4. 평가, 예측

#x_input reshape
x_input = x_input.reshape(1, 3, 1)
y_predict = model.predict(x_input)

loss, acc = model.evaluate(x, y, batch_size=1)

model.summary()

print("y_predict: ",y_predict)
print("loss: ", loss)

'''
Model: "sequential"
_________________________________________________________________ 
Layer (type)                 Output Shape              Param #    
================================================================= 
lstm (LSTM)                  (None, 30)                3840       
_________________________________________________________________ 
dense (Dense)                (None, 70)                2170       
_________________________________________________________________ 
dense_1 (Dense)              (None, 100)               7100       
_________________________________________________________________ 
dense_2 (Dense)              (None, 50)                5050       
_________________________________________________________________ 
dense_3 (Dense)              (None, 30)                1530       
_________________________________________________________________ 
dense_4 (Dense)              (None, 10)                310        
_________________________________________________________________ 
dense_5 (Dense)              (None, 1)                 11
================================================================= 
Total params: 20,011
Trainable params: 20,011
Non-trainable params: 0
_________________________________________________________________ 
y_predict:  [[82.90559]]
loss:  1.5037274360656738
'''