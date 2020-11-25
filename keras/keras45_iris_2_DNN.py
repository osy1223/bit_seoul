# 다중분류

from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

import matplotlib.pyplot as plt
import numpy as np

# 데이터
iris = load_iris()
x = iris.data
y = iris.target

print(y) #0,1,2 라서 3개 
print(x.shape) #(150, 4)
print(y.shape) #(150,)

# 데이터 전처리
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7
)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_predict = x_train[:10]
y_real = y_train[:10]

# 모델링
model = Sequential()
model.add(Dense(500, input_shape=(4,)))
model.add(Dense(20))
model.add(Dense(3, activation='softmax'))

model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 500)               2500
_________________________________________________________________
dense_1 (Dense)              (None, 20)                10020
_________________________________________________________________
dense_2 (Dense)              (None, 3)                 63
=================================================================
Total params: 12,583
Trainable params: 12,583
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
loss :  0.176010861992836
acc :  0.9333333373069763
예측값 :  [1 1 0 0 0 0 2 2 2 0]
실제값 :  [1 1 0 0 0 0 2 2 2 0]
'''