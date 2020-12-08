#1. 데이터
import numpy as np

# input 1개
x1 = np.array((range(1,101), range(711, 811), range(100)))

# output 3개 
y1 = np.array((range(101,201), range(311,411), range(100)))
y2 = np.array((range(501,601), range(431,531), range(100,200)))
y3 = np.array((range(501,601), range(431,531), range(100,200)))

x1=np.transpose(x1)

y1=np.transpose(y1)
y2=np.transpose(y2)
y3=np.transpose(y3)

#train_test_split 2개만 쓰기
from sklearn.model_selection import train_test_split
x1_train, x1_test = train_test_split(
    x1, shuffle=True, train_size=0.7
)

y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(
    y1,y2,y3, shuffle=True, train_size=0.7
)

#2. 함수형 모델 2개 구성

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input

# 모델1
input1 = Input(shape=(3,))
dense1_1 = Dense(10, activation='relu', name='king1')(input1)
dense1_2 = Dense(7, activation='relu', name='king2')(dense1_1)
dense1_3 = Dense(5, activation='relu', name='king3')(dense1_2)
output1 = Dense(3, activation='linear', name='king4')(dense1_3)

#모델 병합, concatenate
# from tensorflow.keras.layers import Concatenate, concatenate

# merge1 = Concatenate()([output1, output2, output3])

middle1 = Dense(30)(output1)
middle2 = Dense(7)(middle1)
middle3 = Dense(11)(middle2)

################# output 모델 구성 (분기)
output1 = Dense(30)(middle3)
output1_1 = Dense(7)(output1)
output1_2 = Dense(3)(output1_1)

output2 = Dense(15)(middle3)
output2_1 = Dense(14)(output2)
output2_3 = Dense(11)(output2_1)
output2_4 = Dense(3)(output2_3)

output3 = Dense(30)(middle3)
output3_1 = Dense(20)(output3)
output3_3 = Dense(10)(output3_1)
output3_4 = Dense(3)(output3_3)

# 모델 정의
model = Model(inputs = input1, 
              outputs = [output1_2, output2_4, output3_4])

model.summary()

# 컴파일 , 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x1_train,
          [y1_train, y2_train, y3_train], epochs=100, batch_size=8,
          validation_split=0.25, verbose=1)

# 평가
result = model.evaluate(x1_test, [y1_test, y2_test, y3_test], 
batch_size=8)

print("result :", result)

'''
result : [3084.46142578125, 
        1007.8184814453125, 
        1052.384765625, 
        1024.258056640625, 
        1007.8184814453125, 
        1052.384765625, 
        1024.258056640625]

RMSE_1:  45.80446268668051
RMSE_2:  32.44402403719858
RMSE_3:  32.58704786288083 

R2_1:  -1.403064429968192
R2_2:  -0.20564446896169666
R2_3:  -0.21629764570312707
'''