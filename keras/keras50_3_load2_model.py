# model load2 (모델+가중치)

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

print(y_train.shape, y_test.shape) # (60000, 10), (10000, 10)

# CNN : 그림의 총 숫자, 가로, 세로, 채널(4차원)

#전처리 /255
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255. 
# x_test.shape[0] 동일
# .astype : 형변환


from tensorflow.keras.models import load_model
model = load_model('./save/model_test01_2.h5')
model.summary()

#모델,컴파일,훈련을 load

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", acc)

# 시각화

'''
save 
loss :  0.06798943132162094
acc :  0.982699990272522

load
loss :  0.06798943132162094
acc :  0.982699990272522
'''