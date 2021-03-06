# diabets Conv1D

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

#1. 데이터
dataset = load_diabetes()

x = dataset.data 
y = dataset.target
print(x.shape) 
print(y.shape) 
# (442, 10)
# (442,)

# 2. 데이터 전처리
# 2.1 OneHotEncoding(회귀모델이라 라벨링 X)
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)


# 2.2 train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7
)


# 2.3 scaler (전체 크기 줄이는 거) 
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print('x_train.shape :',x_train.shape) 
print('x_test.shape :',x_test.shape) 
# x_train.shape : (309, 10)
# x_test.shape : (133, 10)


# 2.4 reshape (데이터 구조 바꾸는 거)
x_train = x_train.reshape(309,10,1)
x_test = x_test.reshape(133,10,1)
print("reshape x:", x_train.shape, x_test.shape)


# 모델링
model = Sequential()
model.add(Conv1D(800, 2, input_shape=(10,1)))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(400))
model.add(Dense(288))
model.add(Dense(150))
model.add(Dense(90))
model.add(Dense(80))
model.add(Dense(1))

model.summary()

# 컴파일, 훈련
model.compile(
    loss='mse', 
    optimizer='adam',
    metrics=['mse'])

es = EarlyStopping(
    monitor='val_loss', 
    patience=7, 
    mode='auto')

cp = ModelCheckpoint(
    filepath='./model/boston_Conv1D_{epoch:02d}_{val_loss:.4f}.hdf5',
    monitor='val_loss',
    save_best_only=True,
    mode='auto')

hist = TensorBoard(
    log_dir='graph', 
    histogram_freq=0,
    write_graph=True, write_images=True)

model.fit(x_train, y_train, 
    epochs=1000, 
    batch_size=32, 
    verbose=1,
    validation_split=0.5, 
    callbacks=[es,cp,hist])

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

print('diabets_Conv1D')

'''
RMSE : 58.06999460355615
R2 :  0.46469701654517415
diabets_Conv1D
'''