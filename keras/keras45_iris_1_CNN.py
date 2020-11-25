# 다중분류 iris cnn

from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.layers import Conv2D, Flatten, MaxPool2D
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
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
x = x.reshape(150, 2, 2, 1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7
)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_predict = x_train[:10]
y_real = y_train[:10]

# 모델링
model = Sequential()
model.add(Conv2D(100, (2,2), padding='same', input_shape=(x_train.shape[1],2,1)))
model.add(Conv2D(300, (1,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(20))
model.add(Dense(3, activation='softmax'))

model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 2, 2, 100)         500
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 2, 2, 300)         30300
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 1, 1, 300)         0
_________________________________________________________________
flatten (Flatten)            (None, 300)               0
_________________________________________________________________
dense (Dense)                (None, 20)                6020
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 63
=================================================================
Total params: 36,883
Trainable params: 36,883
Non-trainable params: 0
_________________________________________________________________
'''

# 컴파일, 훈련
es = EarlyStopping(monitor='loss', patience=10, mode='auto')
hist = TensorBoard(log_dir='graph', histogram_freq=0,
                write_graph=True, write_images=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1,
            validation_split=0.5, callbacks=[es, hist])

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
loss :  0.05286550521850586
acc :  0.9777777791023254
예측값 :  [1 0 1 2 1 0 1 0 2 1]
실제값 :  [1 0 1 2 1 0 1 0 2 1]
'''