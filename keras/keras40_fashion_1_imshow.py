# fashion_mnist

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train[0])
print('y_train[0] : ', y_train[0]) # y_train[0] :  9

print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) #(60000,) (10000,)

plt.imshow(x_train[0], 'gray')
plt.show()