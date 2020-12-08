# y 라벨링 (to_categorical -> OneHotEncoding 필수)
# 모델 아웃풋 레이어에서 'softmax' 사용// 10개 중 하나를 원하니깐 (노드의 개수 맞춰주고)
# 다중분류 컴파일 할때, loss='categorical_crossentropy'
#OneHotEncodeing

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape) # (60000, 28, 28), (10000, 28, 28)
print(y_train.shape, y_test.shape) # (60000,), (10000,)
print(x_train[0])
print(y_train[0])

# plt.imshow(x_train[0], 'gray')
# plt.show()

# 분류방식 : 공평한 가중치 // 분류의 오류에 빠지지 않도록
# 대상인 y(라벨링)를 reshape (60000, 10) 분류 10개 이런식으로

# 데이터 전처리 1.OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape, y_test.shape) # (60000, 10), (10000, 10)
print(y_train[0]) # [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]

# CNN : 그림의 총 숫자, 가로, 세로, 채널(4차원)

#전처리 /255
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255. 
# x_test.shape[0] 동일
# .astype : 형변환
print(x_train[0])

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
# 분산처리 하려고!!
import tensorflow as tf

strategy = tf.distribute.MirroredStrategy(cross_device_ops=\
    tf.distribute.HierarchicalCopyAllReduce()
)

with strategy.scope():
    model = Sequential()
    model.add(Conv2D(10, (2,2), padding='same', input_shape=(28,28,1)))
    model.add(Conv2D(20, (2,2), padding='valid')) # (27, 27, 20)
    model.add(Conv2D(30, (3,3)))
    model.add(Conv2D(40, (2,2), strides=2)) #strides : 옮겨가는 값 (2칸씩 옮길꺼야)
    model.add(MaxPooling2D(pool_size=2)) #pool_size=2 디폴트 값
    model.add(Flatten())
    model.add(Dense(100, activation='relu')) #Dense activation 디폴트 값 : relu
    model.add(Dense(10, activation='softmax')) # 총 결과값 : 10(output)
    # 다중분류 : softmax (2 이상 분류값 이상 중 하나 )

    model.summary()


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
'''
