# 유방암(breast_canceer) 이진 분류

from sklearn.datasets import load_breast_cancer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Dense, Conv2D
from tensorflow.keras.layers import MaxPool2D, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

import matplotlib.pyplot as plt
import numpy as np

# 데이터
cancer = load_breast_cancer()

# 데이터 확인
x = cancer.data #(569, 30)
y = cancer.target #(569, )

print(y) #0,1 // 2개라서 아웃풋 2

# 데이터 전처리
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
x = x.reshape(569,10,3,1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7
)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_predict = x_train[:5]
y_real = y_train[:5]

# 모델링
model = Sequential()
model.add(Conv2D(100, (1,1), input_shape=(10,3,1)))
model.add(Conv2D(200, (2,2)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(20))
model.add(Dense(2, activation='sigmoid'))

model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 10, 3, 100)        200
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 9, 2, 200)         80200
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 4, 1, 200)         0
_________________________________________________________________
flatten (Flatten)            (None, 800)               0
_________________________________________________________________
dense (Dense)                (None, 20)                16020
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 42
=================================================================
Total params: 96,462
Trainable params: 96,462
Non-trainable params: 0
_________________________________________________________________
'''

#컴파일 
es = EarlyStopping(monitor='loss', patience=7, mode='auto')
hist = TensorBoard(log_dir='graph', histogram_freq=0,
        write_graph=True, write_images=True)

model.compile(loss='binary_crossentropy', optimizer='adam',
            metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1,
            validation_split=0.5, callbacks=[es,hist])

#평가, 예측 
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ",loss)
print("acc : ",acc)

y_predict = model.predict(x_predict)
y_predict_recovery = np.argmax(y_predict, axis=1)
y_real = np.argmax(y_real, axis=1)
print('예측값 : ',y_predict_recovery)
print('실제값 : ',y_real) 

'''
loss :  0.27095240354537964
acc :  0.9532163739204407
예측값 :  [1 1 1 1 1]
실제값 :  [1 1 1 1 1]
'''