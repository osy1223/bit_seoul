#cifar10 LSTM

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical #이미지 분류 작업
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.layers import Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard 
import matplotlib.pyplot as plt
import numpy as np

# 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 데이터 전처리
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape) 

#총 이미지 개수, 1개의 이미지의 총 데이터 개수, 몇개씩 연산할지
x_train = x_train.reshape(50000, 32*32*3, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 32*32*3, 1).astype('float32')/255.

x_predict = x_train[:10]
y_real = y_train[:10]
 
# 모델
input1 = Input(shape=(3072, 1))
dense1 = LSTM(10)(input1)
dense2 = Dense(20)(dense1)
dense3 = Dense(30)(dense2)
dense4 = Dense(40)(dense3)
dense5 = Dense(100)(dense4)
output1 = Dense(10, activation='softmax')(dense5)

model = Model(inputs=input1, outputs=output1)

model.summary()

# 컴파일, 훈련
es = EarlyStopping(monitor='loss', patience=5, mode='auto')
to_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True,
                        write_images=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', 
                metrics=['acc'])

model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1,
            validation_split=0.5, callbacks=[es, to_hist])

# 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ",loss)
print("acc : ",acc)

y_predict = model.predict(x_predict)

y_predict_recovery = np.argmax(y_predict, axis=1)
y_real_recovery = np.argmax(y_real, axis=1)

print('실제값 : ',y_real_recovery) 
print('예측값 : ',y_predict_recovery) 
print('cifar10_LSTM')

'''
loss :  2.1223394870758057
acc :  0.22750000655651093
실제값 :  [6 9 9 4 1 1 2 7 8 3]
예측값 :  [7 9 8 6 8 3 4 4 8 7]
'''