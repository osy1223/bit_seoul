# iris Conv1D

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# 데이터
iris = load_iris()
x = iris.data
y = iris.target

print(x.shape) #(150, 4)
print(y.shape) #(150,)

# 2. 데이터 전처리
# 2.1 OneHotEncoding
y = to_categorical(y)
print(y.shape) #(150, 3)

# 2.2 train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7
)

# 2.3 scaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print('x_train.shape :',x_train.shape) #(105, 4)
print('x_test.shape :',x_test.shape) #(45, 4)

# 2.4 reshape (데이터 구조 바꾸는 거)
x_train = x_train.reshape(105,4,1)
x_test = x_test.reshape(45,4,1)
print("reshape x:", x_train.shape, x_test.shape)
# reshape x: (105, 4, 1) (45, 4, 1)


# 모델링
model = Sequential()
model.add(Conv1D(800, 2, input_shape=(4,1)))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(400))
model.add(Dense(288))
model.add(Dense(150))
model.add(Dense(90))
model.add(Dense(80))
model.add(Dense(3, activation='softmax'))

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
    filepath='./model/iris_Conv1D_{epoch:02d}_{val_loss:.4f}.hdf5',
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
print("loss : ",loss)
print("acc : ",acc)

print('iris Conv1D')

'''
loss :  0.090510793030262
acc :  0.9777777791023254
iris Conv1D
'''