#cifar10 CNN

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical #이미지 분류 작업
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard 
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print(x_train.shape, x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3)
# print(y_train.shape, y_test.shape) #(50000, 1) (10000, 1)

# 데이터 전처리 1.OneHotEncoding . 라벨링
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape, y_test.shape) #(50000, 10) (10000, 10)

#전처리 /255
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255. 

x_predict = x_train[:10]
y_real = y_train[:10]

# x_predict = x_predict.reshape(10, 32, 32, 3).astype('float32')/255.

# 모델 구성
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(32,32,3)))
model.add(Conv2D(20, (2,2)))
model.add(Conv2D(30, (2,2)))
model.add(Conv2D(40, (2,2), strides=2))
model.add(MaxPooling2D(pool_size=(2,2))) 
model.add(Flatten()) #3차원을 reshape를 Faltten()으로!
model.add(Dense(100, activation='relu')) 
model.add(Dense(10, activation='softmax'))

model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 31, 31, 10)        130
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 30, 30, 20)        820       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 29, 29, 30)        2430
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 14, 14, 40)        4840
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 7, 7, 40)          0
_________________________________________________________________
flatten (Flatten)            (None, 1960)              0
_________________________________________________________________
dense (Dense)                (None, 100)               196100
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1010
=================================================================
Total params: 205,330
Trainable params: 205,330
Non-trainable params: 0
_________________________________________________________________
'''

# 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', 
                metrics=['acc'])

es = EarlyStopping(monitor='loss', patience=5, mode='auto')
to_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)

model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1, 
            validation_split=0.5, callbacks=[es, to_hist])

#평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ",loss)
print("acc : ",acc)

y_predict = model.predict(x_predict)

y_predict_recovery = np.argmax(y_predict, axis=1)
y_real_recovery = np.argmax(y_real, axis=1)

print('실제값 : ',y_real_recovery) 
print('예측값 : ',y_predict_recovery) 
print('cifar10_CNN')

'''
loss :  4.300537586212158
acc :  0.5968999862670898
실제값 :  [6 9 9 4 1 1 2 7 8 3]
예측값 :  [6 9 9 4 1 1 2 7 8 3]
'''