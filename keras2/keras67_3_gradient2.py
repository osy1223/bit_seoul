import numpy as np
import matplotlib.pyplot as plt

f = lambda x : x**2 - 4*x + 6 #2차원 함수

gradient = lambda x : 2*x - 4 #위에 f 미분한 함수 
# 0이 되는 곳이 최저점이므로 x가 2일때 0이므로 2 찾아가기!

x0 = 0.0 #start 지점
MaxIter = 40
# learning_rate = 0.25
learning_rate = 0.1


print('step\tx\tf(x)')
print('{:02d}\t{:6.5f}\t{:6.5f}'.format(0, x0, f(x0)))
# step    x       f(x)
# 00      0.00000 6.00000

for i in range(MaxIter):
    x1 = x0 - learning_rate * gradient(x0)
    x0 = x1

    print('{:02d}\t{:6.5f}\t{:6.5f}'.format(i+1, x0, f(x0)))

'''
learning_rate = 0.25 일때,
step    x       f(x)
00      0.00000 6.00000
01      1.00000 3.00000
02      1.50000 2.25000
03      1.75000 2.06250
04      1.87500 2.01562
05      1.93750 2.00391
06      1.96875 2.00098
07      1.98438 2.00024
08      1.99219 2.00006
09      1.99609 2.00002
10      1.99805 2.00000

learning_rate = 0.1일때는 10번안에 못찾음
31번째에 나옴 -> 31      1.99802 2.00000
'''