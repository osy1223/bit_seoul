#Dropout

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape) # (60000, 28, 28), (10000, 28, 28)
print(y_train.shape, y_test.shape) # (60000,), (10000,)

# plt.imshow(x_train[0], 'gray')
# plt.show()

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
from tensorflow.keras.layers import Dropout

model = Sequential()
model.add(Conv2D(10, (2,2), padding='same', input_shape=(28,28,1)))
model.add(Dropout(0.2)) #노드 중 80% 쓰겠다. 
#노드의 개수에 20프로를 빼므로, 속도 향상 된다.
model.add(Conv2D(20, (2,2), padding='valid')) 
model.add(Dropout(0.2))
model.add(Conv2D(30, (3,3)))
model.add(Dropout(0.2))
model.add(Conv2D(40, (2,2), strides=2)) #strides : 옮겨가는 값 (2칸씩 옮길꺼야)
model.add(MaxPooling2D(pool_size=2)) #pool_size=2 디폴트 값
model.add(Flatten())
model.add(Dense(100, activation='relu')) #Dense activation 디폴트 값 : relu
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu')) 
model.add(Dropout(0.2))
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
dropout (Dropout)            (None, 28, 28, 10)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 27, 27, 20)        820
_________________________________________________________________
dropout_1 (Dropout)          (None, 27, 27, 20)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 25, 25, 30)        5430
_________________________________________________________________
dropout_2 (Dropout)          (None, 25, 25, 30)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 12, 12, 40)        4840
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 6, 6, 40)          0
_________________________________________________________________
flatten (Flatten)            (None, 1440)              0
_________________________________________________________________
dense (Dense)                (None, 100)               144100
_________________________________________________________________
dropout_3 (Dropout)          (None, 100)               0
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1010
_________________________________________________________________
dropout_4 (Dropout)          (None, 10)                0
_________________________________________________________________
dense_2 (Dense)              (None, 10)                110
=================================================================
Total params: 156,360
Trainable params: 156,360
Non-trainable params: 0
_________________________________________________________________
'''

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# 0~9까지에서 무조건 찾아지는 거니깐, metrics acc 사용
# mse = 'mean_squared_error // acc = 'accuracy 이렇게 풀네임도 가능

model.fit(x_train, y_train, epochs=30, batch_size=32, verbose=1, validation_split=0.5)
# batch_size=32 디폴트값

# 4. 평가, 예측
loss,acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ",loss)
print("acc : ",acc)

'''
val_loss: 0.1379  val_acc: 0.9834
loss :  0.10873852670192719
acc :  0.986299991607666
'''