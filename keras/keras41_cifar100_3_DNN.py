# cifar100 DNN

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout
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

x_train = x_train.reshape(50000, 32*32*3).astype('float32')/255.
x_test = x_test.reshape(10000, 32*32*3).astype('float32')/255.

x_predict = x_train[:10] #(10, 32, 32, 3)

# 모델
model = Sequential()
model.add(Dense(10, input_shape=(32*32*3,)))
model.add(Dense(500, activation='relu')) 
model.add(Dropout(0.2)) 
model.add(Dense(100))
model.add(Dropout(0.2)) 
model.add(Dense(300)) 
model.add(Dense(100))
model.add(Dense(50)) 
model.add(Dense(3)) 
model.add(Dense(100, activation='softmax'))

model.summary()

# 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', 
                metrics=['acc'])

es = EarlyStopping(monitor='loss', patience=1, mode='auto')
to_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)

model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1, 
            validation_split=0.5, callbacks=[es, to_hist])

#평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ",loss)
print("acc : ",acc)

y_predict = model.predict(x_predict)

y_predict_recovery = np.argmax(y_predict, axis=1)

print('실제값 : ',y_real.reshape(10,)) 
print('예측값 : ',y_predict_recovery) 
print('cifar100 DNN')

'''
loss :  3.6389894485473633
acc :  0.14429999887943268
실제값 :  [19 29  0 11  1 86 90 28 23 31]
예측값 :  [89 24 51 52 16 16 16 24 71 24]
'''
