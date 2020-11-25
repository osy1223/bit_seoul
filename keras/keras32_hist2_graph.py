# history 그래프

import numpy as np

dataset = np.array(range(1, 101))
size = 5

# 실습 : 모델 구성
# train, test 분리
# early_stopping

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size +1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset]) #subset
    print(type(aaa))
    return np.array(aaa)

datasets = split_x(dataset, size)
print("=================")
print(datasets)

x = datasets[:, 0:4]
y = datasets[:, 4]

x = np.reshape(x, (x.shape[0], x.shape[1], 1)) #3차원
# x = np.reshape(x, (x.shape[0], x.shape[1], 1, 1)) #4차원 
print(x.shape) # (96, 4, 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=False, train_size=0.9
)

print(x_train.shape) # (86, 4, 1)

# LSTM 함수형 모델 구성
from tensorflow.keras.models import Model , Sequential
from tensorflow.keras.layers import Dense, LSTM, Input

input1 = Input(shape=(4, 1)) 
dense1 = LSTM(100, activation='relu')(input1)
dense2 = Dense(200, activation='relu')(dense1)
dense3 = Dense(300, activation='relu')(dense2)
output = Dense(1)(dense3)

model = Model(inputs=input1, outputs=output)

model.summary()

# 컴파일, 훈련

from tensorflow.keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

history = model.fit(x,y, epochs=100, batch_size=1, verbose=1, validation_split=0.2, callbacks=[early_stopping])

'''
print("-----------------")
print(history)
print("-----------------")
print(history.history.keys())
print("-----------------")
print(history.history['loss'])
print(history.history['val_loss'])
'''
#그래프 
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
#loss라는 y값만 넣어줬다. list형식으로 넣어줬으니 list 형식으로 출력
plt.title('loss & mae')
plt.ylabel('loss, mae')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss', 'train mae', 'val mae']) #legend 명시
plt.show()


