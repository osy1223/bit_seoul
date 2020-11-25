#1. 데이터
import numpy as np

x=np.array((range(1,101), range(311, 411), range(100)))
y=np.array((range(101,201)))


print(x.shape) # (3, 100)
print(y.shape) # (100,)  

x=x.transpose()
print(x.shape) # (100, 3)
# x=x.transpose()로도 가능

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)


#------------- 여기 아래서부터 모델 구성
# y1. y2, y3 = w1x1 + w2x2 + w3x3 +b 

from tensorflow.keras.models import Sequential, Model
#from keras.models import Sequential 로도 가능하나 더 느리다 (API)
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(3,))
dense1 = Dense(5, activation='relu')(input1)
# activation 활성화 함수 (모든 레이어마다 디폴트로 들어가 있다.)
dense2 = Dense(4, activation='relu')(dense1)
dense3 = Dense(3, activation='relu')(dense2)
output = Dense(1)(dense3) #activation='linear'인 상태

model = Model(inputs=input1, outputs=output)

model.summary()

'''
#3. 훈련 
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, validation_split=0.8, verbose=2)

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
'''

