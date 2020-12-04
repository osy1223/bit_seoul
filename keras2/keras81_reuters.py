from tensorflow.keras.datasets import reuters

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. 데이터
(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2
)
# num_words 에서 잘 나오는 단어 개수 지정해줌

print(x_train.shape, x_test.shape) # (8982,) (2246,)
print(y_train.shape, y_test.shape) # (8982,) (2246,) 

# print(x_train[0])
# print(y_train[0])

# print(len(x_train[0])) #87
# print(len(x_train[11])) #59

# y의 카테고리 개수 출력
category = np.max(y_train)+1
print('카테고리 종류:', category) #카테고리 종류: 46

# y의 유니크한 값들 출력
y_bunpo = np.unique(y_train)
# print(y_bunpo)
'''
카테고리 종류: 46
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
'''

# 실습
# embedding 모델구성해서 돌리기

from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, maxlen=100, padding='pre') 
x_test = pad_sequences(x_test, maxlen=100, padding='pre')
# maxlen으로 크기 잘라서 가져오기 // 최대값으로 가져오면 너무 길어요 ㅠ_ㅠ
print(x_train.shape) #(8982, 100)
print(x_test.shape) #(2246, 100)

# 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten

model = Sequential()
model.add(Embedding(10000, 50, input_length=100))
model.add(LSTM(32))
model.add(Dense(46, activation='softmax'))
model.summary()

'''
sparse_categorical_entropy
별도 원핫 인코딩을 하지 않고 정수값 그대로 줄 수 있다.
'''

# 3. 컴파일 , 훈련
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

model.fit(x_train, y_train, epochs=30)

# 4. 평가
acc = model.evaluate(x_test, y_test)[1]
print('acc:',acc)

'''
acc: 0.6798753142356873
'''
