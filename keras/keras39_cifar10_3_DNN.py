#'cifar10_DNN'

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical #이미지 분류 작업
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard 
import matplotlib.pyplot as plt
import numpy as np

# 데이터 불러오기
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# y 라벨링
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

'''
print(y_train.shape, y_test.shape) 
(50000, 10) (10000, 10)
print(x_train.shape, x_test.shape) 
(50000, 32, 32, 3) (10000, 32, 32, 3)
'''

#데이터 scaling, reshape
x_train = x_train.reshape(50000, 32*32*3).astype('float32')/255.
x_test = x_test.reshape(10000, 3072).astype('float32')/255.

# 데이터 10개만 가져오기
x_predict = x_train[:10]
y_real = y_train[:10]

# 모델 구성
model = Sequential()
model.add(Dense(10, input_shape=(3072,)))
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(400))
model.add(Dense(1000))
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(10, activation='softmax'))

model.summary()


# 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['acc'])

es = EarlyStopping(monitor='loss', patience=5, mode='auto')
hist = TensorBoard(log_dir='graph', histogram_freq=0,
                write_graph=True, write_images=True)

model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1,
            validation_split=0.5, callbacks=[es, hist])

# 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ",loss)
print("acc : ",acc)

y_predict = model.predict(x_predict)

y_predict_recovery = np.argmax(y_predict, axis=1)

y_real_recovery = np.argmax(y_real, axis=1)

print('실제값 : ',y_real_recovery) 
print('예측값 : ',y_predict_recovery) 
print('cifar10_DNN')

'''
loss :  1.786475419998169
acc :  0.3569999933242798
실제값 :  [6 9 9 4 1 1 2 7 8 3]
예측값 :  [6 1 9 6 8 7 4 7 8 9]
'''