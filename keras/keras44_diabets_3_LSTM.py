#dataset diabets

import numpy as np
from sklearn.datasets import load_diabetes
dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x.shape) #(442, 10)
print(y.shape) #(442,)
#컬럼 10개 y까지하면 총 11개

# 데이터 전처리
# train, test 분리. validation
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

x = x.reshape(442,10,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7
)

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

model = Sequential()
model.add(LSTM(10, input_shape=(10,1)))
model.add(Dense(300))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

model.summary()

# 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard 
model.compile(loss='mse', optimizer='adam', 
                metrics=['mse'])

es = EarlyStopping(monitor='loss', patience=100, mode='auto')
to_hist = TensorBoard(log_dir='graph', histogram_freq=0,
                    write_graph=True, write_images=True)

model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=1, 
            validation_split=0.5, callbacks=[es, to_hist])


# 평가, 예측
y_predict = model.predict(x_test)

# RMSE, R2
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE :", RMSE(y_test,y_predict))

# R2 함수
from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)
print("R2 : ",r2)

print('dataset diabets LSTM')

'''
RMSE : 63.616911308879025
R2 :  0.22870151752834766
'''