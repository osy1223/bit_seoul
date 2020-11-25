# cifar100 CNN

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard 

import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, x_test.shape) 
#(50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape) 
#(50000, 1) (10000, 1)

y_real = y_train[:10]
print(y_real)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

x_predict = x_train[:10] #(10, 32, 32, 3)

# 모델
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(32, 32, 3)))
model.add(Conv2D(20, (2,2)))
model.add(Conv2D(30, (2,2)))
model.add(Conv2D(50, (2,2)))
model.add(Conv2D(30, (2,2)))
model.add(Conv2D(40, (2,2), strides=2))
model.add(MaxPooling2D(pool_size=(2,2))) 
model.add(Flatten()) #3차원을 reshape를 Faltten()으로!
model.add(Dense(50, activation='relu')) 
model.add(Dense(100, activation='softmax'))

model.summary()

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


print('실제값 : ',y_real.reshape(10,)) 
print('예측값 : ',y_predict_recovery) 
print('cifar100 CNN')

'''
loss :  13.160635948181152
acc :  0.2029000073671341
실제값 :  [19 29  0 11  1 86 90 28 23 31]
예측값 :  [19 29  0 11  1 86 90 28 23 31]
'''
