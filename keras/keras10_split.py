#validation_split

#1. 데이터
import numpy as np

x=np.array(range(1,101))
y=np.array(range(101,201))

#70개(1~70)
x_train = x[:71]
y_train = y[:71]

#30개(71~100)
x_test = x[71:]   
y_test = y[71:]

#나머지 코드를 완성하시오.
#2.모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(30,input_dim=1))
model.add(Dense(100))
model.add(Dense(5000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

#3. 컴파일
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

#훈련 validation_split(0.2)
model.fit(x_train, y_train, epochs=100, validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
print("결과물 : ",y_predict)

#RMSE 함수 사용자 정의
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE :", RMSE(y_test,y_predict))

#R2 함수
from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)
print("R2 : ",r2)
