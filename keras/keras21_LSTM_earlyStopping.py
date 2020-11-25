#실습 LSTM , earlyStopping

# 1. 데이터

import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
            [5,6,7], [6,7,8], [7,8,9], [8,9,10],
            [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x_input = np.array([50,60,70])

# shape 확인
print("x.shape : ", x.shape) #x.shape :  (13, 3)
print("y.shape : ", y.shape) #y.shape :  (13,)


# 함수형 모델
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM

input1 = Input(shape=(3,1))
dense1 = LSTM(100, activation='relu')(input1)
dense2 = Dense(200, activation='relu')(dense1)
dense3 = Dense(200, activation='relu')(dense2)
dense4 = Dense(200, activation='relu')(dense3)
dense5 = Dense(200, activation='relu')(dense4)
output1 = Dense(1)(dense5)

model = Model(inputs=input1, outputs=output1)

model.summary()

# 3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping #조기종료

#early stopping이 무조건 좋지 않음! hyper parameter tuning 부분 추가!
# early_stopping = EarlyStopping(monitor='loss', patience=100, mode='min')
# mode = min 최소값
# 감시 기준을 loss
# 최소값보다 내려가면 계속진행, 올라간다면 멈춤
# patience 몇번까지 봐줄거냐? 바로 끝내기보다는 조금 더 지켜보고 최소값을 정하겠다. 
# min/max 헷갈릴 때 auto
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')

model.fit(x, y, epochs=1000, batch_size=1, verbose=1, callbacks=[early_stopping])


# 4. 평가, 예측
x_input = x_input.reshape(1, 3, 1)
predict = model.predict(x_input)
print("predict :", predict)
