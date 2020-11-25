#Dropout cifar10 CNN

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard 

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, x_test.shape) 
#(50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape) 
#(50000, 1) (10000, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#전처리 /255
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255. 

# 모델
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(32,32,3)))
model.add(Dropout(0.2))
model.add(Conv2D(30, (3,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()

# 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
            metrics=['acc'])

es = EarlyStopping(monitor='loss', patience=5, mode='auto')
to_hist = TensorBoard(log_dir='graph', histogram_freq=0, 
                    write_graph=True, write_images=True)

model.fit(x_train, y_train, epochs=30, batch_size=32, verbose=1,
            validation_split=0.5)

# 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ",loss)
print("acc : ",acc)

'''
val_loss: 1.5343 - val_acc: 0.5667
loss :  1.5203797817230225
acc :  0.5655999779701233
'''