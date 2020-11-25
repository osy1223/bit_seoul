#Dropout mnist CNN

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape) 
# (60000, 28, 28), (10000, 28, 28)
print(y_train.shape, y_test.shape) 
# (60000,), (10000,)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#전처리 /255
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255. 

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

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# 0~9까지에서 무조건 찾아지는 거니깐, metrics acc 사용

model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.5)
# batch_size=32 디폴트값

# 4. 평가, 예측
loss,acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ",loss)
print("acc : ",acc)

'''
val_loss: 0.1104 - val_acc: 0.9796
loss :  0.08010796457529068
acc :  0.9829999804496765
'''