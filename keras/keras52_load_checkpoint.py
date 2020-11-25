# checkpoint load

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape) # (60000, 28, 28), (10000, 28, 28)
print(y_train.shape, y_test.shape) # (60000,), (10000,)
print(x_train[0])
print(y_train[0])

# 데이터 전처리 1.OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#전처리 /255
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255. 

# 2. 모델
# 3. 컴파일, 훈련

from tensorflow.keras.models import load_model
model = load_model('./model/minist-05-0.0730.hdf5')


# 4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", result[0])
print("acc : ", result[1])

'''
loss :  0.10326086729764938
acc :  0.9812999963760376

loss :  0.060174841433763504
acc :  0.9828000068664551
'''

#체크포인트가 더 좋다. loss 보십시오~