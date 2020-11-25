# mnist2_CNN 실습

'''
실습1. test데이터를 10개 가져와서 predict 만들 것
    - 원핫 인코딩을 원복할 것
    print('실제값 : ', 어쩌구 저쩌구) #결과 : [3 4 5 2 9 1 3 9 0]
    print('예측값 : ', 어쩌구 저쩌구) #결과 : [3 4 5 2 9 1 3 9 1]


실습2. 모델: es 적용, tensorboard도 넣을 것
'''

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape) # (60000, 28, 28), (10000, 28, 28)
print(y_train.shape, y_test.shape) # (60000,), (10000,)
print(x_train[0])
print(y_train[0])

x_predict = x_test[:10]
y_real = y_test[:10]
# 데이터 전처리 1.OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape, y_test.shape) # (60000, 10), (10000, 10)
# print(y_train[0]) # [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]



# CNN : 그림의 총 숫자, 가로, 세로, 채널(4차원)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255. 

x_predict=x_predict.reshape(10,28,28,1).astype('float32')/255.
# .astype : 형변환

# print(x_train[0])

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Conv2D(10, (2,2), padding='same', input_shape=(28,28,1)))
model.add(Conv2D(20, (2,2), padding='valid')) # (27, 27, 20)
model.add(Conv2D(50, (3,3)))
model.add(Conv2D(40, (2,2), strides=2)) #strides : 옮겨가는 값 (2칸씩 옮길꺼야)
model.add(MaxPooling2D(pool_size=2)) #pool_size=2 디폴트 값
model.add(Flatten())
model.add(Dense(100, activation='relu')) #Dense activation 디폴트 값 : relu
model.add(Dense(10, activation='softmax')) # 총 결과값 : 10(output)
# 다중분류 : softmax (2 이상 분류값 이상 중 하나 )

# model.summary()

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard 

es = EarlyStopping(monitor='loss', patience=100, mode='auto')
to_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(x_train,y_train, epochs=1, batch_size=32, 
                verbose=1, validation_split=0.2, callbacks=[es, to_hist])

# 4. 평가, 예측
loss,acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ",loss)
print("acc : ",acc)

# np.argmax 함수를 사용하면 인코딩된 데이터를 역으로 되돌릴 수 있다.
y_predict = model.predict(x_predict)
y_predict_recovery = np.argmax(y_predict, axis=1).reshape(-1,1)

print('실제값 : ',y_real)
print('예측값 : ',y_predict_recovery)

'''
loss :  0.0763792023062706
acc :  0.9769999980926514
실제값 :  [7 2 1 0 4 1 4 9 5 9]
예측값 :  [[7]
 [2]
 [1]
 [0]
 [4]
 [1]
 [4]
 [9]
 [5]
 [9]]
'''
