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
from tensorflow.keras.layers import Dense, LSTM, GRU

model = Sequential()
model.add(GRU(10, activation='relu', input_shape=(3,1)))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
gru (GRU)                    (None, 10)                390       
_________________________________________________________________
dense_112 (Dense)            (None, 20)                220       
_________________________________________________________________
dense_113 (Dense)            (None, 10)                210       
_________________________________________________________________
dense_114 (Dense)            (None, 1)                 11        
=================================================================
Total params: 831
Trainable params: 831
Non-trainable params: 0
_________________________________________________________________
'''


