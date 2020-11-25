# cancer Conv1D

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# 1. 데이터 로드
cancer = load_breast_cancer()

x = cancer.data #(569, 30)
y = cancer.target #(569, )

print("y value category:",set(y))
# y value category: {0, 1}

# 2. 데이터 전처리
# 2.1 OneHotEncoding
# 이진분류라서 free~ free~

# 2.2 train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7
)


# 2.3 scaler (전체 크기 줄이는 거) 
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print('x_train.shape :',x_train.shape) 
print('x_test.shape :',x_test.shape) 
# x_train.shape : (398, 30)
# x_test.shape : (171, 30)


# 1.4 reshape (데이터 구조 바꾸는 거)
x_train = x_train.reshape(398, 30, 1)
x_test = x_test.reshape(171, 30, 1)
print("reshape x:", x_train.shape, x_test.shape)


# 모델링
model = Sequential()
model.add(Conv1D(500, 2, input_shape=(30,1)))
model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dense(200))
model.add(Dense(150))
model.add(Dense(100))
model.add(Dense(90))
model.add(Dense(80))
model.add(Dense(1, activation='sigmoid')) #이진분류 output1 

model.summary()

# 컴파일, 훈련
model.compile(
    loss='binary_crossentropy', 
    optimizer='adam',
    metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss', 
    patience=7, 
    mode='auto')

cp = ModelCheckpoint(
    filepath='./model/boston_Conv1D_{epoch:02d}_{val_loss:.4f}.hdf5',
    monitor='val_loss',
    save_best_only=True,
    mode='auto')

hist = TensorBoard(
    log_dir='graph', 
    histogram_freq=0,
    write_graph=True, write_images=True)

model.fit(x_train, y_train, 
    epochs=1000, 
    batch_size=33, 
    verbose=1,
    validation_split=0.2, 
    callbacks=[es,cp,hist])

#평가, 예측 
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ",loss)
print("acc : ",acc)


'''
loss :  0.12323693931102753
acc :  0.9941520690917969
'''