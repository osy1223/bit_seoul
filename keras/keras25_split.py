#이 소스를 분석하시오.

import numpy as np
dataset = np.array(range(1, 11))
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size +1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset]) #subset
    print(type(aaa))
    return np.array(aaa)

datasets = split_x(dataset, size)
print("=================")
print(datasets)

'''
([item for item in subset]) 변경 

<class 'list'>
=================
[[ 1  2  3  4  5]
 [ 2  3  4  5  6]
 [ 3  4  5  6  7]
 [ 4  5  6  7  8]
 [ 5  6  7  8  9]
 [ 6  7  8  9 10]]

1~10까지의 데이터를 5씩 잘라서 
 
 <class 'list'>
=================
       x      | y
[[ 1  2  3  4 | 5] 
 [ 2  3  4  5 | 6]
 [ 3  4  5  6 | 7]
 [ 4  5  6  7 | 8]
 [ 5  6  7  8 | 9]
 [ 6  7  8  9 | 10]]

x, y로 만들어 주는 함수

'''
