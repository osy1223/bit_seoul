#1. 데이터
import numpy as np

x=np.array((range(1,101)))
y=np.array((range(101,201), range(711,811), range(100)))

print(x.shape) #(100, )
print(y.shape) #(3,100)


y = y.transpose()
print("-----------")
print(y.shape) #(100,3)
print(x.shape) #(100, )

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)

# ------------- 여기 아래서부터 모델 구성
# # y1. y2, y3 = w1x1 + w2x2 + w3x3 +b 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=1)) #input 1개
model.add(Dense(5))
model.add(Dense(700))
model.add(Dense(5000))
model.add(Dense(10000))
model.add(Dense(400))
model.add(Dense(3)) #output 3개


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