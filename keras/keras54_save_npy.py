from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
print(iris)
print(type(iris))
# 'filename': 'C:\\Anaconda3\\lib\\site-packages\\sklearn\\datasets\\data\\iris.csv'}
# <class 'sklearn.utils.Bunch'>

x_data = iris.data
y_data = iris.target

print(type(x_data))
print(type(y_data))
# <class 'numpy.ndarray'>
# <class 'numpy.ndarray'>

#numpy 파일은 앞으로 data 폴더에 저장할겁니다

np.save('./data/iris_x.npy', arr=x_data)
np.save('./data/iris_y.npy', arr=y_data)
