# 2. 실습 return_sequences

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

x_input = x_input.reshape(1,3,1)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

'''
LSTM 2개 엮을 때 Error
ValueError: Input 0 of layer lstm_1 is incompatible with the layer: 
expected ndim=3, found ndim=2. Full shape received: [None, 30]   
# input과 output의 shape가 맞지 않을 때 나타나는 오류
3차원에서 2차원으로 넘겨지니깐 error, 3차원으로 넘겨줘야해  

_________________________________________________________________   
Layer (type)                 Output Shape              Param #      
=================================================================   
lstm (LSTM)                  (None, 30)                3840
_________________________________________________________________   
dense (Dense)                (None, 1)                 31
=================================================================   
Total params: 3,871
Trainable params: 3,871
Non-trainable params: 0
_________________________________________________________________  
2차원

_________________________________________________________________   
Layer (type)                 Output Shape              Param #      
=================================================================   
lstm (LSTM)                  (None, 3, 30)             3840
_________________________________________________________________   
dense (Dense)                (None, 3, 1)              31
=================================================================   
Total params: 3,871
Trainable params: 3,871
Non-trainable params: 0
_________________________________________________________________  
3차원

'''
#시계열 데이터였다면 두 개가 더 나은 성능을 보여 줬을 수도 있다 
#case by case이므로 직접 다 경험해 봐야 
#데이터의 구조나 모델에 따라서 다르다 
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(3, 1), return_sequences=True)) 
#output이 dense layer에 맞게 2차원 (None, 30)으로 나가게 됨
model.add(LSTM(180, activation='relu')) #단 문제가 있음. 잘라서 쓸 수 없음.
model.add(Dense(150, activation='relu'))
model.add(Dense(110, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()

# # 3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')

# from tensorflow.keras.callbacks import EarlyStopping #조기종료

# early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')
# model.fit(x, y, epochs=1000, batch_size=1, verbose=1, callbacks=[early_stopping])

# # 4. 평가, 예측
# predict = model.predict(x_input)
# print("predict :", predict)

# loss = model.evaluate(x_input, np.array([80]), batch_size=1)
# print("loss :", loss)
