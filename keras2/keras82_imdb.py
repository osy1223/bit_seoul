from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 소스를 완성하시오. embedding

# 1. 데이터
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000
)
print(x_train.shape, x_test.shape) # (25000,) (25000,)
print(y_train.shape, y_test.shape) # (25000,) (25000,)

# print(len(x_train[0])) #218
# print(len(x_train[11])) #99

# y의 카테고리 개수 출력
category = np.max(y_train)+1
# print('카테고리 종류:', category) # 카테고리 종류: 2

# y의 유니크한 값들 출력
y_bunpo = np.unique(y_train)
# print(y_bunpo) #[0 1] 2진분류


len_result = [len(s) for s in x_train]
# print('리뷰의 최대 길이:{}'.format(np.max(len_result)))
# print('리뷰의 평균 길이:{}'.format(np.mean(len_result)))
'''
리뷰의 최대 길이:2494
리뷰의 평균 길이:238.71364
'''

# 시각화로 확인
plt.subplot(1,2,1)
plt.boxplot(len_result)
plt.subplot(1,2,2)
plt.hist(len_result, bins=50)
# plt.show()

from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, maxlen=100, padding='pre')
x_test = pad_sequences(x_test, maxlen=100, padding='pre')

print(x_train.shape) #(25000, 100)
print(x_test.shape) #(25000, 100)

# 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten

model = Sequential()
model.add(Embedding(10000, 50, input_length=100))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# 3. 컴파일 , 훈련
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['acc']
)

model.fit(x_train, y_train, epochs=30)

# 4. 평가
acc = model.evaluate(x_test, y_test)[1]
print('acc:',acc)

'''
acc: 0.818120002746582
'''