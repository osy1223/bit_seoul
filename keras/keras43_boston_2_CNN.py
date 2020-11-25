#dataset boston CNN

import numpy as np
from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target

print(x.shape) #(506, 13)
print(y.shape) #(506,)

# 데이터 전처리
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

x = x.reshape(506,13,1,1)
# train, test 분리. validation
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7
)

# CNN
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D
from tensorflow.keras.layers import MaxPooling2D, Flatten

model = Sequential()
model.add(Conv2D(10, (1,1), input_shape=(13,1,1), padding='same'))
model.add(Conv2D(100, (1,1), padding='valid'))
model.add(MaxPooling2D(pool_size=1))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(1))

model.summary()

# 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard 
model.compile(loss='mse', optimizer='adam', 
                metrics=['mse'])

es = EarlyStopping(monitor='loss', patience=5, mode='auto')
to_hist = TensorBoard(log_dir='graph', histogram_freq=0,
                    write_graph=True, write_images=True)

model.fit(x_train, y_train, epochs=300, batch_size=10, verbose=1, 
            validation_split=0.2, callbacks=[es, to_hist])


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

print('boston_CNN')

'''
RMSE : 5.445771335163489
R2 :  0.6619517936316661
'''