# cifar10의 최적화 튠으로 구성하시오!

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split 
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("x_train.shape:", x_train.shape) #x_train.shape: (50000, 32, 32, 3)
print("x_test.shape:", x_test.shape) #x_test.shape: (10000, 32, 32, 3)
print("y_test.shape:", y_test.shape) #y_test.shape: (10000, 1)

# OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) # (50000, 10) (10000, 10)
# print(y_train[0]) #[0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]

# CNN을 위한 reshape
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3])
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_train.shape[3])
print("reshape x:", x_train.shape, x_test.shape)

# 2. 모델 
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3])))
model.add(Conv2D(20, 3))
model.add(Conv2D(10, 2))
model.add(Flatten())
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))
model.summary()

# 3. 컴파일, 훈련
model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam',
    metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=6
)

modelpath = './model/cancer-{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(
    filepath=modelpath,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss',
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    patience=3,
    factor=0.5,
    verbose=1
)

model.fit(x_train, y_train, 
    epochs=30, 
    batch_size=32, 
    verbose=1, 
    validation_split=0.5,
    callbacks=[es, cp]
)

# 평가, 예측
loss,acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ",loss)
print("acc : ",acc)


'''
loss :  2.634603261947632
acc :  0.3084000051021576
'''