# 데이터 전처리, MinMaxScaler

from numpy import array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input

# 1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
            [5,6,7], [6,7,8], [7,8,9], [8,9,10],
            [9,10,11], [10,11,12],
            [2000,3000,4000],[3000,4000,5000],[4000,5000,6000],
            [100,200,300]])  

# x = 훈련시키기 위한 데이터

y = array([4,5,6,7,8,9,10,11,12,13,5000,6000,7000,400]) # y : (14,)
# y = label, target
#전처리에서 y는 target이라서 건드리지 않고, x만 줄여서 데이터 정제
#x[1,2,3] - y[4], x[100,200,300] - y[400] 이렇게 매칭 

x_predict1 = array([55,65,75]) 
x_predict2 = array([6600,6700,6800])

x_predict1 = x_predict1.reshape (1,3)
x_predict2 = x_predict2.reshape (1,3)

#데이터 전처리 위한 Min, Max 스케일
from sklearn.preprocessing import MinMaxScaler 

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
x_predict1 = scaler.transform(x_predict1)
x_predict2 = scaler.transform(x_predict2)

'''
print(x)
[[0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [2.50062516e-04 2.00080032e-04 1.66750042e-04]
 [5.00125031e-04 4.00160064e-04 3.33500083e-04]
 [7.50187547e-04 6.00240096e-04 5.00250125e-04]
 [1.00025006e-03 8.00320128e-04 6.67000167e-04]
 [1.25031258e-03 1.00040016e-03 8.33750208e-04]
 [1.50037509e-03 1.20048019e-03 1.00050025e-03]
 [1.75043761e-03 1.40056022e-03 1.16725029e-03]
 [2.00050013e-03 1.60064026e-03 1.33400033e-03]
 [2.25056264e-03 1.80072029e-03 1.50075038e-03]
 [4.99874969e-01 5.99839936e-01 6.66499917e-01]
 [7.49937484e-01 7.99919968e-01 8.33249958e-01]
 [1.00000000e+00 1.00000000e+00 1.00000000e+00]
 [2.47561890e-02 3.96158463e-02 4.95247624e-02]]
'''
'''
print('x_predict :', x_predict)
x_predict : [[0.01350338 0.01260504 0.012006  ]]

print('x_predict2 :', x_predict2)
x_predict2 : [[1.65016254 1.34013605 1.13340003]]
'''

# 모델 부터 ~
# predict도 scale 변환해야 합니다~

# LSTM 함수형 모델 구성
from tensorflow.keras.models import Model , Sequential
from tensorflow.keras.layers import Dense, LSTM, Input

x = x.reshape(14,3,1)
x_predict1 = x_predict1.reshape(1,3,1)
x_predict2 = x_predict2.reshape(1,3,1)

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3,1)))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

# 컴파일, 훈련

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, shuffle=False, train_size=0.8
# )

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard 

es = EarlyStopping(monitor='loss', patience=100, mode='auto')
to_hist = TensorBoard(log_dir='graph', histogram_freq=0, 
                    write_graph=True, write_images=True)

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

history = model.fit(x,y, epochs=10, batch_size=1, 
                verbose=1, callbacks=[es, to_hist])

# 예측, 훈련

y_predict1 = model.predict(x_predict1)
print("y_predict1 :", y_predict1)

y_predict2 = model.predict(x_predict2)
print("y_predict2 :", y_predict2)