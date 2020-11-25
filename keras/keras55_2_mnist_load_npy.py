# np.load 

import numpy as np
import matplotlib.pyplot as plt
# from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# 데이터
x_train = np.load('./data/mnist_x_train.npy')
x_test = np.load('./data/mnist_x_test.npy')
y_train = np.load('./data/mnist_y_train.npy')
y_test = np.load('./data/mnist_y_test.npy')

print(x_train.shape, x_test.shape) 
print(y_train.shape, y_test.shape)


# 데이터 전처리 1. OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape, y_test.shape)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.


#2. 모델
################### 1. load_model ########################

#3. 컴파일, 훈련
from tensorflow.keras.models import load_model
model1 = load_model('./save/model_test02_2.h5')

#4. 평가, 예측
result1 = model1.evaluate(x_test, y_test, batch_size=32)
print("r1 loss : ", result1[0])
print("r1 acc : ", result1[1])


############## 2. load_model ModelCheckPoint #############
from tensorflow.keras.models import load_model
model2 = load_model('./model/minist-05-0.0738.hdf5')

#4. 평가, 예측
result2 = model2.evaluate(x_test, y_test, batch_size=32)
print("r2 loss : ", result2[0])
print("r2 accuracy : ", result2[1])

############### 3. load_weights ##################
# 2. 모델
model3 = Sequential()
model3.add(Conv2D(10, (2,2), padding='same', input_shape=(28,28,1)))
model3.add(Conv2D(20, (2,2), padding='valid'))
model3.add(Conv2D(30, (3,3)))
model3.add(Conv2D(40, (2,2), strides=2))
model3.add(MaxPooling2D(pool_size=2))
model3.add(Flatten())
model3.add(Dense(100, activation='relu'))
model3.add(Dense(10, activation='softmax'))

# 3. 컴파일
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model3.load_weights('./save/weight_test02_3.h5')

#4. 평가, 예측
result3 = model3.evaluate(x_test, y_test, batch_size=32)
print("r3 loss : ", result3[0])
print("r3 acc : ", result3[1])

'''
r1 loss :  0.09647425264120102
r1 acc :  0.9837999939918518

r2 loss :  0.058727119117975235
r2 accuracy :  0.9833999872207642

r3 loss :  0.09647425264120102
r3 acc :  0.9837999939918518
'''