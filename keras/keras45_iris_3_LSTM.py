# 다중분류

from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Dense, LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

import matplotlib.pyplot as plt
import numpy as np

# 데이터
iris = load_iris()
x = iris.data
y = iris.target

print(x.shape) #(150, 4)
print(y.shape) #(150,)

# 데이터 전처리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7
)

# scaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print('x_train:',x_train.shape) #(105, 4)
print('x_test:',x_test.shape) #(45, 4)

# categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_predict = x_train[:10]
y_real = y_train[:10]

# 모델링
model = Sequential()
model.add(LSTM(10, input_shape=(4,1)))
model.add(Dense(300))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(3, activation='softmax'))

model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 10)                480
_________________________________________________________________
dense (Dense)                (None, 300)               3300
_________________________________________________________________
dropout (Dropout)            (None, 300)               0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               30100
_________________________________________________________________
dense_2 (Dense)              (None, 200)               20200
_________________________________________________________________
dense_3 (Dense)              (None, 100)               20100
_________________________________________________________________
dense_4 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_5 (Dense)              (None, 3)                 153
=================================================================
Total params: 79,383
Trainable params: 79,383
Non-trainable params: 0
_________________________________________________________________
'''

# 컴파일, 훈련
es = EarlyStopping(monitor='loss', patience=10, mode='auto')
hist = TensorBoard(log_dir='graph', histogram_freq=0,
                write_graph=True, write_images=True)

model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['acc'])
model.fit(x_train, y_train, 
    epochs=100, 
    batch_size=32, 
    verbose=1,
    validation_split=0.5, 
    callbacks=[es, hist])

# 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ",loss)
print("acc : ",acc)

y_predict = model.predict(x_predict)
y_predict_recovery = np.argmax(y_predict, axis=1)
y_real = np.argmax(y_real, axis=1)
print('예측값 : ',y_predict_recovery)
print('실제값 : ',y_real) 

'''
loss :  0.155710831284523
acc :  0.9555555582046509
예측값 :  [0 2 0 2 2 1 2 1 2 1]
실제값 :  [0 2 0 2 2 1 2 1 2 1]
'''