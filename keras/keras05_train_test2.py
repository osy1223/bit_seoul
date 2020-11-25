import numpy as np

#1. 데이터 준비
x_train=np.array([1,2,3,4,5,6,7,8,9,10])
y_train=np.array([1,2,3,4,5,6,7,8,9,10])
#훈련하고 싶은건 _train
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
#평가,테스트 하고 싶은건 _test
x_pred=np.array([16,17,18])
#예측할 데이터 _pred (y_pred는 알고 싶은것)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델 구성
model = Sequential()
model.add(Dense(30,input_dim=1))
model.add(Dense(500))
model.add(Dense(200))
model.add(Dense(800))
model.add(Dense(900))
model.add(Dense(200))
model.add(Dense(700))
model.add(Dense(1))

#3. 컴파일, 훈련 (컴퓨터가 알아들을 수 있도록)
# model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
#훈련
model.fit(x_train, y_train, epochs=100)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)

print("loss : ", loss)

y_pred = model.predict(x_pred)
print("결과물 : ",y_pred)

#실습 : 결과물 오차 수정. 미세조정
