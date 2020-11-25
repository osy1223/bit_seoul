# 나머지 6개 dataset을 저장하시오

from sklearn.datasets import load_boston
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_diabetes
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100

import numpy as np

# load_data, data type

#---------------------------------boston
#<class 'sklearn.utils.Bunch'>

boston = load_boston()
print(type(boston))

boston_x = boston.data
boston_y = boston.target

print(type(boston_x))
print(type(boston_y))

np.save('./data/boston_x.npy', arr=boston_x)
np.save('./data/boston_y.npy', arr=boston_y)

#---------------------------------cancer
cancer = load_breast_cancer()
print(type(cancer))

cancer_x = cancer.data
cancer_y = cancer.target

np.save('./data/cancer_x.npy', arr=cancer_x)
np.save('./data/cancer_y.npy', arr=cancer_y)

#---------------------------------diabets
diabets = load_diabetes()
print(type(diabets))

diabets_x = diabets.data
diabets_y = diabets.target

np.save('./data/diabets_x.npy', arr=diabets_x)
np.save('./data/diabets_y.npy', arr=diabets_y)

#---------------------------------fashion
# <class 'tuple'>

(fm_x_train, fm_y_train), (fm_x_test, fm_y_test) = fashion_mnist.load_data()
print(type(fashion_mnist))
# <class 'module'>

print(fm_x_train.shape, fm_x_test.shape)  #(60000, 28, 28) (10000, 28, 28)
print(fm_y_train.shape, fm_y_test.shape)  #(60000,) (10000,)

np.save('./data/fm_x_train.npy', arr = fm_x_train)
np.save('./data/fm_x_test.npy', arr = fm_x_test)
np.save('./data/fm_y_train.npy', arr = fm_y_train)
np.save('./data/fm_y_test.npy', arr = fm_y_test)


'''
AttributeError: 'tuple' object has no attribute 'data'
'''

#---------------------------------cifar10
(c10_x_train, c10_y_train), (c10_x_test, c10_y_test) = cifar10.load_data()
print(type(cifar10))

np.save('./data/c10_x_train.npy', arr = c10_x_train)
np.save('./data/c10_x_test.npy', arr = c10_x_test)
np.save('./data/c10_y_train.npy', arr = c10_y_train)
np.save('./data/c10_y_test.npy', arr = c10_y_test)

#---------------------------------cifar100
(c100_x_train, c100_y_train), (c100_x_test, c100_y_test) = cifar100.load_data()
print(type(cifar100))

np.save('./data/c100_x_train.npy', arr = c100_x_train)
np.save('./data/c100_x_test.npy', arr = c100_x_test)
np.save('./data/c100_y_train.npy', arr = c100_y_train)
np.save('./data/c100_y_test.npy', arr = c100_y_test)