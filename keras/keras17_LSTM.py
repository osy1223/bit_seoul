# LSTM

# 1. 데이터
import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]]) # x.shape:(4, 3)
y = np.array([4,5,6,7]) # y.shape:(4,)

print("x.shape : ", x.shape)
print("y.shape : ", y.shape)

x = x.reshape(x.shape[0], x.shape[1], 1)
# x = x.reshape(4, 3, 1)
# 위의 2가지 중 1가지 쓰시면 됩니다.
print("x.shape : ", x.shape)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3,1)))
#LSTM에서는 몇개씩 작업할지 명시해줘야 한다. 
#DNN 1차원? // LSTM : 3차원 shape 필요. 행,열,몇개씩 자르는지 (?,?,?)
model.add(Dense(40))
model.add(Dense(90))
model.add(Dense(50))
model.add(Dense(90))
model.add(Dense(200))
model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='mse')

model.fit(x, y, epochs=100, batch_size=1)
x_input = np.array([5, 6, 7]) #(3, ) -> (1, 3, 1) #LSTM에 들어가려면 맞춰 줘야 
#데이터 reshape
x_input = x_input.reshape(1, 3, 1)
#어느 정도 데이터 양도 필요하다 (LSTM 할 때)


#4. 평가, 예측
y_predict = model.predict(x_input)
loss, acc = model.evaluate(x, y, batch_size=1)

print("예측값: ", y_predict)
print("loss: ", loss, "\n", "acc: ", acc)

# model.summary()
# LSTM parameter = 4 * (몇개씩잘라서 + 1 (bias) + node수) * node수
#LSTM은 그만큼 연산이 많다 따라서 속도도 그만큼 느려짐

# LSTM은 input으로 2차원을 받는다 (DNN은 행 빼고 스칼라만 받고)
# LSTM의 shape => (행, 열, 몇 개씩 자르는지) 단, 전체 데이터 개수는 같게 reshape
# y값은 대부분 스칼라로 많이 나감. 그런데 LSTM은 parameter가 되게 많다(순환)
# 하지만 데이터 개수 적으니까 잘라서 하나씩! (3, 1) 즉, 1열씩 작업하겠다

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 10)                480
_________________________________________________________________
dense (Dense)                (None, 40)                440
_________________________________________________________________
dense_1 (Dense)              (None, 90)                3690
_________________________________________________________________
dense_2 (Dense)              (None, 50)                4550
_________________________________________________________________
dense_3 (Dense)              (None, 90)                4590
_________________________________________________________________
dense_4 (Dense)              (None, 200)               18200
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 201
=================================================================
Total params: 32,151
Trainable params: 32,151
Non-trainable params: 0
_________________________________________________________________

예측값:  [[7.9884853]]
loss:  2.271481935167685e-05
acc:  2.271481935167685e-05

'''