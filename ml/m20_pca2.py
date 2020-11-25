import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes


# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape) #(442, 10)
print(y.shape) #(442,)

'''
# 1.1 데이터 전처리
pca = PCA(n_components=4) #n_components : 줄일 차원의 수
x2d = pca.fit_transform((x)) #fit_transform : 한번에 fit+transform
print(x2d.shape) #(442, 4)
 
pca_EVR = pca.explained_variance_ratio_ 
print(pca_EVR) #n_components의 개수(4개)만큼 출력 됩니다.
print(sum(pca_EVR))
'''

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)

print(cumsum)
# [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759
#  0.94794364 0.99131196 0.99914395 1.        ] 
# 차원 축소 몇개 할지 판단 기준이 된다. 손실되고 남은 값

d = np.argmax(cumsum >= 0.95) +1
print(cumsum >= 0.95)
# [False False False False False False False  True  True  True]
print(d)
# 8

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()
