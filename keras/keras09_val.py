import numpy as np

#1. 데이터 준비
x_train=np.array([1,2,3,4,5,6,7,8,9,10])
y_train=np.array([1,2,3,4,5,6,7,8,9,10])
x_val = np.array([11,12,13,14,15])
y_val = np.array([11,12,13,14,15])
# x_pred=np.array([16,17,18])
x_test = np.array([16,17,18,19,20])
y_test = np.array([16,17,18,19,20])


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

#훈련할때 훈련용 데이터, 검증용 데이터 (문제지-답안지 컨닝)
model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val))


#4. 평가, 예측
loss = model.evaluate(x_test,y_test)

print("loss : ", loss)

y_predict = model.predict(x_test)
print("결과물 : ",y_predict)

#실습 : 결과물 오차 수정. 미세조정

#RMSE 함수 사용자 정의
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE :", RMSE(y_test,y_predict))

#R2 함수
from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)
print("R2 : ",r2)