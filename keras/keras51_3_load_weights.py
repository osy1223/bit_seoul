# cp load

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape) # (60000, 28, 28), (10000, 28, 28)
print(y_train.shape, y_test.shape) # (60000,), (10000,)

# 데이터 전처리 1.OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#전처리 /255
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255. 
# x_test.shape[0] 동일
# .astype : 형변환

# 2. 모델

# 3. 컴파일, 훈련
from tensorflow.keras.models import load_model
model = load_model('./model/minist-05-0.0738.hdf5')
model.compile(loss='categorical_crossentropy', optimizer='adam', 
                metrics=['acc'])



# 4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", result[0])
print("acc : ", result[1])

'''
save
loss :  0.09647425264120102
acc :  0.9837999939918518

load weight
loss :  0.09647425264120102
acc :  0.9837999939918518

load cp
loss :  0.058727119117975235
acc :  0.9833999872207642
'''