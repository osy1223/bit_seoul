import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.arange(-5, 5, 1)
y = sigmoid(x)

print('x:', x)
print('y:', y)

'''
x: [-5 -4 -3 -2 -1  0  1  2  3  4]
y: [0.00669285 0.01798621 0.04742587 0.11920292 0.26894142 0.5
 0.73105858 0.88079708 0.95257413 0.98201379]
'''

# 시각화
plt.plot(x,y)
plt.grid()
plt.show()