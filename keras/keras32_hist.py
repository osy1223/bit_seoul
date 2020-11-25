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

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from tensorflow.keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')

history = model.fit(x,y, epochs=10, batch_size=10, verbose=1, validation_split=0.2, callbacks=[early_stopping])

print("-----------------")
print(history)
'''
<tensorflow.python.keras.callbacks.History object at 0x00000251B12FD880>
'''
print("-----------------")
print(history.history.keys())
'''
dict_keys(['loss', 'mse', 'val_loss', 'val_mse'])  
->loss와 metrics 
'''

print("-----------------")
print(history.history['loss'])
print(history.history['val_loss'])

'''
[230.57725524902344, 1.7599256038665771, 0.25352299213409424, 0.11556178331375122, 0.05331813171505928, 0.031105583533644676, 0.026820145547389984, 0.015675866976380348, 0.014550809748470783, 0.015427862294018269]
[29.580944061279297, 3.3389499187469482, 0.4902350902557373, 0.0024612261913716793, 0.021593963727355003, 0.0004128411819692701, 0.0014162075240164995, 0.09300459921360016, 0.02390459179878235, 0.002272141631692648]
10개씩 출력. 왜?  epochs=10 라서!
'''

