# 유방암(breast_canceer) 이진 분류

from sklearn.datasets import load_breast_cancer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

import matplotlib.pyplot as plt
import numpy as np

# 데이터
cancer = load_breast_cancer()

# 데이터 확인
x = cancer.data #(569, 30)
y = cancer.target #(569, )

print(y) #0,1 // 2개라서 아웃풋 2

# 데이터 전처리

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7
)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


x_predict = x_train[:5]
y_real = y_train[:5]

# 모델링
model = Sequential()
model.add(Dense(400, input_shape=(30,)))
model.add(Dense(20))
model.add(Dense(2, activation='sigmoid'))

model.summary()

#컴파일 
es = EarlyStopping(monitor='loss', patience=7, mode='auto')
hist = TensorBoard(log_dir='graph', histogram_freq=0,
        write_graph=True, write_images=True)

model.compile(loss='binary_crossentropy', optimizer='adam',
            metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1,
            validation_split=0.5, callbacks=[es,hist])

#평가, 예측 
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ",loss)
print("acc : ",acc)

y_predict = model.predict(x_predict)
y_predict_recovery = np.argmax(y_predict, axis=1)
y_real = np.argmax(y_real, axis=1)
print('예측값 : ',y_predict_recovery)
print('실제값 : ',y_real) 

'''
loss :  0.09477243572473526
acc :  0.9590643048286438
예측값 :  [0 0 1 1 1]
실제값 :  [0 0 1 1 1]
'''

'''
y는 0 또는 1

1. 원핫인코딩

2. 모델링 마지막 output
Dense(2, activation='sigmoid')
이진분류라도 분류니깐 

평가지표 metrics = acc로 넣기

3. 카테고리컬 크로스 엔트로피 (CCE)
이진분류는 binary_crossentropy
'''