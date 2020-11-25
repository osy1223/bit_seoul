# 실습 : 앙상블 함수형 모델 구현 
# 예상값 : 85 (첫번째 모델에 가중치)

from numpy import array

# 1. 데이터 (input 2개 output 1개)
x1 = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
            [5,6,7], [6,7,8], [7,8,9], [8,9,10],
            [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])

x2 = array([[10,20,30], [20,30,40], [30,40,50], [40,50,60],
            [50,60,70], [60,70,80], [70,80,90], [80,90,100],
            [90,100,110], [100,110,120],
            [2,3,4], [3,4,5], [4,5,6]])

y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x1_predict = array([55,65,75]) #85
x2_predict = array([65,75,85]) #95

print("x1.shape : ", x1.shape) #x1.shape :  (13, 3)
print("x2.shape : ", x2.shape) #x2.shape :  (13, 3)
print("y.shape : ", y.shape)   #y.shape :  (13,) 스칼라가 13개 (shape가 다르다)


# 3차원 reshape
x1 = x1.reshape(13, 3, 1)
x2 = x2.reshape(13, 3, 1)

x1_predict = x1_predict.reshape(1, 3, 1)
x2_predict = x2_predict.reshape(1, 3, 1)
# 리스트 1개의 데이터, 요소 1개씩 총 3개를 1개씩 연산 잘라서 사용


# 2. 함수형 모델
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM

# x1 모델
input1 = Input(shape=(3,1)) #요소는 총3개고 1씩 잘라서 사용할거다 ~ 데이터의 갯수는 필요없다~
dense1_1 = LSTM(200, activation='relu')(input1)
dense1_2 = Dense(180, activation='relu')(dense1_1)
dense1_3 = Dense(100, activation='relu')(dense1_2)
dense1_4 = Dense(50, activation='relu')(dense1_3)
dense1_5 = Dense(10, activation='relu')(dense1_4)
output1 = Dense(1)(dense1_5)
# model1 = Model(inputs=input1, outputs=output1)

# x2 모델
input2 = Input(shape=(3,1))
dense2_1 = LSTM(200, activation='relu')(input2)
dense2_2 = Dense(180, activation='relu')(dense2_1)
dense2_3 = Dense(100, activation='relu')(dense2_2)
dense2_4 = Dense(50, activation='relu')(dense2_3)
dense2_5 = Dense(10, activation='relu')(dense2_4)
output2 = Dense(1)(dense2_5)
# model2 = Model(inputs=input2, outputs=output2)

# 모델 병합
from tensorflow.keras.layers import Concatenate

merge1 = Concatenate()([output1, output2])

middle1 = Dense(50)(merge1)
middle1 = Dense(70)(middle1)
middle1 = Dense(50)(middle1)

# output 모델
output3 = Dense(10)(middle1)
output3_1 = Dense(10)(output3)
output3_2 = Dense(10)(output3_1)
output3_3 = Dense(1)(output3_2)

# 모델 정의
model = Model(inputs = [input1, input2], 
                outputs= output3_3)

model.summary()

# 3. 컴파일, 훈련

#early stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=85, mode='auto')

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit([x1, x2], y, epochs=100, batch_size=1, verbose=1)

# 4. 예측, 평가

y_predict1 = model.predict([x1_predict, x2_predict])

print(y_predict1)


