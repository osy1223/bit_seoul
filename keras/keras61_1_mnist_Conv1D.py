# mnist Conv1D

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint

# 1. 데이터 로드
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# mnist는 분류형 모델입니다. 

print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) #(60000,) (10000,)

# 2. 데이터 전처리
# 2.1 OneHotEncoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape, y_test.shape) #(60000, 10) (10000, 10)

# 2.2 train_test_split
# train과 test가 미리 나눠져 있으니 별도로 하지 않는다
# validation은 fit에서 별도로 split 한다

# 2.3 scaler
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

# 2.4 reshape
x_train = x_train.reshape(60000, 28*28,1)
x_test = x_test.reshape(10000, 28*28,1)
print("reshape x:", x_train.shape, x_test.shape)
# reshape x: (60000, 784, 1) (10000, 784, 1)

# 모델링
model = Sequential()
model.add(Conv1D(100, 2, input_shape=(28*28,1)))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10, activation='softmax'))

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
    filepath='./model/mnist_Conv1D_{epoch:02d}_{val_loss:.4f}.hdf5',
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
loss,acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ",loss)
print("acc : ",acc)

'''
loss :  0.18262961506843567
acc :  0.9635000228881836
'''


# 1.데이터
# 1.1 load_data
# 1.2 train_test_split
# 1.3 scaler
# 1.4 reshape
# 2.모델
# 3.컴파일 훈련
# 4.평가 예측
