# 모델 save

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 2. 모델
model = Sequential()
model.add(LSTM(100, input_shape=(3,1)))
model.add(Dense(50, name='queen1'))
model.add(Dense(10, name='queen2'))
# model.add(Dense(1, name='queen3'))

# model.summary()

model.save("./save/keras28.h5") #.은 root 작업 폴더 , h5 확장자
#model.save(".\save\keras28_2.h5") # \n 이 개행이므로, 개행으로 받아들여서 오류 
# model.save(".//save//keras28_3.h5")
# model.save(".\\save\\keras28_4.h5")

'''
ValueError: 
Weights for model sequential have not yet been created.
Weights are created when the Model is first called on inputs or `build()` is called with an `input_shape`.  
'''