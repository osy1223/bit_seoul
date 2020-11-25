#cifar10

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical #이미지 분류 작업
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train[0])
print('y_train[0] : ', y_train[0]) #y_train[0] :  [6]

print(x_train.shape, x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3) #5만장의 데이터가 32픽셀
print(y_train.shape, y_test.shape) #(50000, 1) (10000, 1) #테스트가 1만장

plt.imshow(x_train[0])
plt.show()