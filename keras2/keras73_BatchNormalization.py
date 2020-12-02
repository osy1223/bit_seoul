# y 라벨링 (to_categorical -> OneHotEncoding 필수)
# 모델 아웃풋 레이어에서 'softmax' 사용// 10개 중 하나를 원하니깐 (노드의 개수 맞춰주고)
# 다중분류 컴파일 할때, loss='categorical_crossentropy'

#OneHotEncodeing

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 분류방식 : 공평한 가중치 // 분류의 오류에 빠지지 않도록
# 대상인 y(라벨링)를 reshape (60000, 10) 분류 10개 이런식으로

# 데이터 전처리 1.OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# CNN : 그림의 총 숫자, 가로, 세로, 채널(4차원)

#전처리 /255
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255. 
# x_test.shape[0] 동일
# .astype : 형변환

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import BatchNormalization, Activation, Dropout
from tensorflow.keras.regularizers import l1, l2, l1_l2

model = Sequential()
model.add(Conv2D(10, (2,2), padding='same', input_shape=(28,28,1)))

model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(20, (2,2), kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(30, (3,3), kernel_regularizer=l1(l1=0.01)))
model.add(Dropout(0.2))


model.add(Conv2D(20, (2,2), padding='valid')) # (27, 27, 20)
model.add(Conv2D(30, (3,3)))
model.add(Conv2D(40, (2,2), strides=2)) #strides : 옮겨가는 값 (2칸씩 옮길꺼야)
model.add(MaxPooling2D(pool_size=2)) #pool_size=2 디폴트 값
model.add(Flatten())
model.add(Dense(100, activation='relu')) #Dense activation 디폴트 값 : relu
model.add(Dense(10, activation='softmax')) # 총 결과값 : 10(output)
# 다중분류 : softmax (2 이상 분류값 이상 중 하나 )

model.summary()

'''
Model: "sequential"
_________________________________________________________________        
Layer (type)                 Output Shape              Param #
=================================================================        
conv2d (Conv2D)              (None, 28, 28, 10)        50
_________________________________________________________________        
conv2d_1 (Conv2D)            (None, 27, 27, 20)        820
_________________________________________________________________        
conv2d_2 (Conv2D)            (None, 25, 25, 30)        5430
_________________________________________________________________        
conv2d_3 (Conv2D)            (None, 12, 12, 40)        4840
_________________________________________________________________        
max_pooling2d (MaxPooling2D) (None, 6, 6, 40)          0
_________________________________________________________________        
flatten (Flatten)            (None, 1440)              0
_________________________________________________________________        
dense (Dense)                (None, 100)               144100
_________________________________________________________________        
dense_1 (Dense)              (None, 10)                1010
=================================================================        
Total params: 156,250
Trainable params: 156,250
Non-trainable params: 0
_________________________________________________________________        
PS D:\Study>
'''

# 3. 컴파일, 훈련
model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam',
    metrics=['acc'])

# 0~9까지에서 무조건 찾아지는 거니깐, metrics acc 사용
# mse = 'mean_squared_error // acc = 'accuracy 이렇게 풀네임도 가능

model.fit(x_train, y_train, 
    epochs=30, 
    batch_size=32, 
    verbose=1, 
    validation_split=0.5)
# batch_size=32 디폴트값

# 4. 평가, 예측
loss,acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ",loss)
print("acc : ",acc)

'''
loss :  0.08234968781471252
acc :  0.9854999780654907


BatchNomalization, kernel_initializer, kernel_regularizer 사용 후
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 28, 28, 10)        50
_________________________________________________________________
batch_normalization (BatchNo (None, 28, 28, 10)        40
_________________________________________________________________
activation (Activation)      (None, 28, 28, 10)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 27, 27, 20)        820
_________________________________________________________________
batch_normalization_1 (Batch (None, 27, 27, 20)        80
_________________________________________________________________
activation_1 (Activation)    (None, 27, 27, 20)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 25, 25, 30)        5430
_________________________________________________________________
dropout (Dropout)            (None, 25, 25, 30)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 24, 24, 20)        2420
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 22, 22, 30)        5430
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 11, 11, 40)        4840
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 5, 5, 40)          0
_________________________________________________________________
flatten (Flatten)            (None, 1000)              0
_________________________________________________________________
dense (Dense)                (None, 100)               100100
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1010
=================================================================
Total params: 120,220
Trainable params: 120,160
Non-trainable params: 60
_________________________________________________________________

loss :  0.12837950885295868
acc :  0.9847999811172485
'''
