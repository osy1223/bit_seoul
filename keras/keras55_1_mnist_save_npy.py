# np.save

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape)  # (60000,28,28), (10000,28,28)
print(y_train.shape, y_test.shape)  # (60000, )      (10000, )

np.save('./data/mnist_x_train.npy', arr = x_train)
np.save('./data/mnist_x_test.npy', arr = x_test)
np.save('./data/mnist_y_train.npy', arr = y_train)
np.save('./data/mnist_y_test.npy', arr = y_test)

