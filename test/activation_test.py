import numpy as np
import matplotlib.pyplot as plt
 
plt.figure()
x = np.arange(-5, 5, 0.01)
 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
 
plt.plot(x, sigmoid(x), linestyle = '--',label = "Sigmoid")
 
def tanh(x):
    return np.tanh(x)
 
plt.plot(x, tanh(x), linestyle = '--', label = "tanh")
 
def relu(x):
    return (x > 0) * x
 
plt.plot(x, relu(x), linestyle = '--', label = "Relu")
 
def Leaky_relu(x):
    return (x >= 0) * x + (x < 0) * x * 0.01
 
plt.plot(x, Leaky_relu(x),  linestyle = '--', label = "Leaky Relu")
 
def ELU(x):
    return (x >= 0) * x + (x < 0) * (np.exp(x)-1)
 
plt.plot(x, ELU(x),  linestyle = '--', label = "ELU")
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.grid()
plt.legend()
plt.show()
