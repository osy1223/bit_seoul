# a02_ae.py 카피
# 딥하게 구성 // CNN으로 구성
# 인코더 부분을 cnn 2~3개로 구성
# padding = same과 valid 비교하십시오

import numpy as np
from tensorflow.keras.datasets import mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, _), (x_test, _) = mnist.load_data()
# print(x_train.shape) #(60000, 28, 28) 
# print(x_test.shape) #(10000, 28, 28)


x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1)/255.

# print(x_train[0])
# print(x_test[0])
# print(x_train.shape) #(60000, 28, 28, 1)
# print(x_test.shape) #(10000, 28, 28, 1)


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, LSTM

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size, 
                    kernel_size=(1,1), 
                    input_shape=(28, 28, 1), 
                    activation='relu'))
    model.add(Flatten())
    model.add(Dense(units=20))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

model = autoencoder(hidden_layer_size=154)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# model.compile(optimizer='adam', loss='mse', metrics=['acc'])

model.fit(x_train, x_train.reshape(60000,784), epochs=10)

output = model.predict(x_test)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = \
    plt.subplots(2, 5, figsize=(20, 7))

# 이미지 5개를 무작위로 고른다
random_image = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_image[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_image[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()


