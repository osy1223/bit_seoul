#  데이터 전처리 개념으로 pca 사용!

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# 1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()
print(x_train.shape) #(60000, 28, 28)
print(x_test.shape) #(10000, 28, 28)

x = np.append(x_train, x_test, axis=0)
print(x.shape) #(70000, 28, 28)

## 실습
# pca를 통해 마구마구  0.95 이상인게 몇개?
# pca를 배운거 다 집어넣고 확인

# 1.1 데이터 전처리
x = x.reshape(70000, 28*28)
print(x.shape) #(70000, 784)

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
# print(cumsum)

d = np.argmax(cumsum >= 1) +1 #d : 우리가 필요한 n_components의 개수
# print(cumsum >= 0.95)
print(d) 
# 713

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()

pca = PCA(n_components=713)
x2d = pca.fit_transform((x))
print(x2d.shape) #(70000, 713)

pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)
print(sum(pca_EVR)) #1.0000000000000024