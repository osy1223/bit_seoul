# GRU

import numpy as np

# 1. 데이터
import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]]) # x.shape:(4, 3)
y = np.array([4,5,6,7]) # y.shape:(4,)

print("x.shape : ", x.shape)
print("y.shape : ", y.shape)

x = x.reshape(x.shape[0], x.shape[1], 1)
# x = x.reshape(4, 3, 1)
# 위의 2가지 중 1가지 쓰시면 됩니다.
print("x.shape : ", x.shape)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN

model = Sequential()
model.add(SimpleRNN(10, activation='relu', input_shape=(3,1)))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

'''
_________________________________________________________________   
Layer (type)                 Output Shape              Param #      
=================================================================   
simple_rnn (SimpleRNN)       (None, 10)                120
_________________________________________________________________   
dense (Dense)                (None, 20)                220
_________________________________________________________________   
dense_1 (Dense)              (None, 10)                210
_________________________________________________________________   
dense_2 (Dense)              (None, 1)                 11
=================================================================   
Total params: 561
Trainable params: 561
Non-trainable params: 0
_________________________________________________________________ 
'''


