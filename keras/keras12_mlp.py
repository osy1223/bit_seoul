#1. 데이터
import numpy as np

x=np.array((range(1,101), range(311, 411), range(100)))
y=np.array((range(101,201), range(711,811), range(100)))

print(x[1][10]) #321
print(y[1][10]) #721
print(x.shape) #(3, 100)
print(y.shape) #(3, 100)

#(100, 3)의 형태로 변환해서 사용해야 한다
#내일부턴 이걸로 모델링 할꺼야 ~_~

#실습 (100,3) 100행 3열 데이터셋을 만들어 내시오. 3행 100열 
print(np.array(x))
print(np.array(y))

x=np.transpose(x)
y=np.transpose(y)
# x.transpose()로도 가능

print(x.shape) #(100, 3)
print(y.shape) #(100, 3)

print(np.array(x))
print(np.array(y))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)
#------------- 여기 위에까지 데이터 구축 완성



#------------- 여기 아래서부터 모델 구성
# y1. y2, y3 = w1x1 + w2x2 + w3x3 +b 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(5))
model.add(Dense(3)) #출력 컬럼 3개



#---------------------나머지 완성
#3. 컴파일
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

#훈련 
model.fit(x_train, y_train, epochs=100, validation_split=0.3)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
print("결과물 : ", y_predict)

#RMSE 함수 사용자 정의
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE :", RMSE(y_test,y_predict))

#R2 함수
from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)
print("R2 : ",r2)

print("x_test : ", x_test)