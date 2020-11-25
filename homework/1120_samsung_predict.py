import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# np.save('./data/samsung.npy', arr=samsung)
# 1. samsung numpy 파일 불러오기 (allow_pickle=True)
samsung = np.load('./data/samsung.npy', allow_pickle=True)
# print(samsung)
# print(samsung.shape) #(620, 5)
'''
[[50200 50400 49100 49200 18709146]
 [49200 50200 49150 49850 15918683]
 [50300 50500 49400 49400 10365440]
 ...
 [67000 67000 65600 65700 30204089]
 [65700 66200 64700 64800 22963790]
 [64100 64800 63900 64600 16590290]]
'''

# 2. x축, y축 자르기

'''
def split_x(seq, size):
    aaa = [] # 임시 리스트
    # i는 0부터 seq사이즈-size까지 반복 
    # (그래야 size만큼씩 온전히 자를 수 있다)
    for i in range(len(seq) -size +1 ):
        subset = seq[i:(i+size)] # subset은 i부터 size만큼 배열 저장
        aaa.append([subset]) # 배열에 subset을 붙인다
    print(type(aaa)) # aaa의 타입은 리스트
    return np.array(aaa) # 리스트를 어레이로 바꿔서 반환하자
'''


def split_xy(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(dataset):
            break
        tem_x = dataset[i:x_end_number, :]
        tem_y = dataset[x_end_number:y_end_number, 3]

        x.append(tem_x)
        y.append(tem_y)
    return np.array(x), np.array(y)

x, y = split_xy(samsung, 5, 1)
print(samsung)
print(x, "\n", y)

# 3.데이터 전처리 
print(x.shape)
print(y.shape)
'''
(615, 5, 5)
(615, 1)
'''

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = True, train_size=0.7)

print(x_train.shape) #(430, 5, 5)
print(x_test.shape) #(185, 5, 5)

# reshape
# ValueError: Found array with dim 3. StandardScaler expected <= 2
x_train = x_train.reshape(430, 5*5)
x_test = x_test.reshape(185, 5*5)

print(x_train.shape) #(430, 25)
print(x_test.shape) #(185, 25)

# scaler 
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train[0,:])

# 4. samsung 모델 만들기
model = Sequential()
model.add(Dense(64, input_shape=(25,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 64)                1664
_________________________________________________________________
dense_1 (Dense)              (None, 32)                2080
_________________________________________________________________
dense_2 (Dense)              (None, 32)                1056
_________________________________________________________________
dense_3 (Dense)              (None, 32)                1056
_________________________________________________________________
dense_4 (Dense)              (None, 32)                1056
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 33
=================================================================
Total params: 6,945
Trainable params: 6,945
Non-trainable params: 0
_________________________________________________________________
'''