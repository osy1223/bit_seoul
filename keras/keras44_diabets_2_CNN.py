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
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
x = x.reshape(442,5,2,1)

# train, test 분리. validation
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7
)

# CNN 모델
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D
from tensorflow.keras.layers import MaxPooling2D, Flatten

model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(5,2,1), padding='same'))
model.add(Conv2D(100, (4,1), padding='valid'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(500))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dropout(0.3))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(1))

model.summary()

# 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard 

es = EarlyStopping(monitor='loss', patience=10, mode='auto')
to_hist = TensorBoard(log_dir='graph', histogram_freq=0,
                    write_graph=True, write_images=True)

model.compile(loss='mse', optimizer='adam', 
                metrics=['mse'])

model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1, 
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

print('dataset diabets CNN')

'''
RMSE : 64.43983025823675
R2 :  0.38400453132879253
'''