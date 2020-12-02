import numpy as np
import matplotlib.pyplot as plt

# relu 함수
def relu(x):
    return np.maximum(0, x)

x = np.arange(-5, 5, 0.1)
y = relu(x)

plt.plot(x,y)
plt.grid()
plt.show()

# relu 친구들 찾기
# elu, selu, ??elu???

# 이진 함수
def binary_step_activate_function(x):
    y = x>0
    return y.astype(np.int)

x = np.arange(-5, 5, 0.1)
y = binary_step_activate_function(x)

plt.plot(x,y)
plt.show()

# 리키 킬루 함수
a = 0.1
def leaky_relu(x):
    return np.maximum(a*x, x)

x = np.arange(-5, 5, 0.1)
y = leaky_relu(x)

plt.plot(x,y)
plt.grid()
plt.show()


#######################################################
# ReLU계열
#######################################################

# ReLU(Rectified Linear Unit, 정류된 선형 유닛) 함수
def relu_func(x): 
    return (x>0)*x
    # return np.maximum(0,x) # same
 
#그래프 출력
plt.plot(x, relu_func(x), label="ReLU")
plt.grid()
plt.show()
 
# Leaky ReLU(Rectified Linear Unit, 정류된 선형 유닛) 함수
def leakyrelu_func(x): 
    return (x>=0)*x + (x<0)*0.01*x # 알파값(보통 0.01) 조정가능
    # return np.maximum(0.01*x,x) # same
 
#그래프 출력
plt.plot(x, leakyrelu_func(x), label="Leaky ReLU")
plt.grid()
plt.show()
 
# ELU(Exponential linear unit)
def elu_func(x):
    return (x>=0)*x + (x<0)*0.01*(np.exp(x)-1)
 
#그래프 출력
plt.plot(x, elu_func(x),  label="ELU(Exponential linear unit)")
plt.grid()
plt.show()

# Thresholded ReLU
def trelu_func(x): # Thresholded ReLU
    return (x>1)*x # 임계값(1) 조정 가능
 
#그래프 출력
plt.plot(x, trelu_func(x),  label="Thresholded ReLU")
plt.grid()
plt.show()
 