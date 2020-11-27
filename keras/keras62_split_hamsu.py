# split 함수 커스마이징

# iris 데이터를
# 150,4를 5개씩 자른다면 146,5,4가 되어야 한다

# 데이터셋을 자르는 함수 -> 이게 자격증 5번 문제

import numpy as np

def split_x(seq, size):
    aaa = [] # 임시 리스트
    # i는 0부터 seq사이즈-size까지 반복 
    # (그래야 size만큼씩 온전히 자를 수 있다)
    for i in range(len(seq) -size +1):
        subset = seq[i:(i+size)] # subset은 i부터 size만큼 배열 저장
        aaa.append([subset]) # 배열에 subset을 붙인다
    print(type(aaa)) # aaa의 타입은 리스트
    return np.array(aaa) # 리스트를 어레이로 바꿔서 반환하자

# 데이터
dataset = np.array([1,2,3,4,5,6,7,8,9,10])

# (다 : 1) split 함수
def split_xy1(dataset, time_steps):
    x, y = list(), list()
    for i in range(len(dataset)):
        end_number = i + time_steps
        if end_number > len(dataset) -1:
            break
        tmp_x, tmp_y = dataset[i:end_number], dataset[end_number]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy1(dataset,4)
print('x:', x)
print('y:', y)
'''
x: [[1 2 3 4]
 [2 3 4 5]
 [3 4 5 6]
 [4 5 6 7]
 [5 6 7 8]
 [6 7 8 9]]
y: [ 5  6  7  8  9 10]
'''

# (다 : 다) split 함수
def split_xy2(dataset, time_steps, y_column)
