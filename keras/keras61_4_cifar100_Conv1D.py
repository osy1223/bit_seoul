# cifar100 Conv1D

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint

# 1.데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, x_test.shape) 
# (50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape)
# (50000, 1) (10000, 1)

# 2. 데이터 전처리
# 2.1 OneHotEncoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape, y_test.shape)
# (50000, 100) (10000, 100)

# 2.2 train_test_split
# train과 test가 미리 나눠져 있으니 별도로 하지 않는다
# validation은 fit에서 별도로 split 한다


# 1.3 scaler (전체 크기 줄이는 거) (함수는 2차원밖에 안되서 수동으로 줄인겁니다!)
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

# 1.4 reshape (데이터 구조 바꾸는 거)
x_train = x_train.reshape(50000, 32*32, 3)
x_test = x_test.reshape(10000, 32*32, 3)
print("reshape x:", x_train.shape, x_test.shape)


# 모델링
model = Sequential()
model.add(Conv1D(50, 2, input_shape=(32*32, 3)))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(100, activation='softmax'))

model.summary()

# 컴파일, 훈련
model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam',
    metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss', 
    patience=7, 
    mode='auto')

cp = ModelCheckpoint(
    filepath='./model/cifar10_Conv1D_{epoch:02d}_{val_loss:.4f}.hdf5',
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
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", acc)
print('cifar100_conv1D')
'''
loss :  4.259828090667725
acc :  0.1501999944448471
cifar100_conv1D
'''