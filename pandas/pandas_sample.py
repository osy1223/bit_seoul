import pandas as pd
import numpy as np

from numpy.random import randn
np.random.seed(100)

data = randn(5,4)
print(data)
'''
[[-1.74976547  0.3426804   1.1530358  -0.25243604]    
 [ 0.98132079  0.51421884  0.22117967 -1.07004333]    
 [-0.18949583  0.25500144 -0.45802699  0.43516349]    
 [-0.58359505  0.81684707  0.67272081 -0.10441114]    
 [-0.53128038  1.02973269 -0.43813562 -1.11831825]]
    (5,4)
'''

df = pd.DataFrame(data,
    index='A B C D E'.split(),
    columns='가 나 다 라'.split())
print(df)
'''
          가         나         다         라
A -1.749765  0.342680  1.153036 -0.252436
B  0.981321  0.514219  0.221180 -1.070043
C -0.189496  0.255001 -0.458027  0.435163
D -0.583595  0.816847  0.672721 -0.104411
E -0.531280  1.029733 -0.438136 -1.118318
컬럼(헤더) : 가 나 다 라, 인덱스 : A B C D E
'''

data2 = [[1,2,3,4],
        [5,6,7,8],
        [9,10,11,12],
        [13,14,15,16],
        [17,18,19,20]]
        #5행 4열
df2 = pd.DataFrame(data2,
    index=['A', 'B', 'C', 'D', 'E'],
    columns=['가', '나', '다', '라'])
print(df2)
'''
    가   나   다   라
A   1   2   3   4
B   5   6   7   8
C   9  10  11  12
D  13  14  15  16
E  17  18  19  20
'''

df3 = pd.DataFrame(np.array([[1,2,3],[4,5,6]]))
print(df3)
'''
   0  1  2
0  1  2  3
1  4  5  6
자동으로 넣어주는 인덱스와 헤더는 데이터가 아니다
'''

#컬럼
print("df2['나']:\n", df2['나']) #2,6,10,14,18
'''
df2['나']:
 A     2
B     6
C    10
D    14
E    18
Name: 나, dtype: int64
'''
print("df2['나', '라']:\n", df2[['나','라']]) 
#2, 6, 10, 14, 18 // 4, 8, 12, 16, 20
'''
df2['나', '라']:
     나   라
A   2   4
B   6   8
C  10  12
D  14  16
E  18  20
'''

# print("df2[0]",df2[0])
# KeyError: 0 -> 컬럼명으로 해줘야 에러 안납니다

# print("df2.loc['나']\n:", df2.loc['나'])
# KeyError: '나' loc는 행에서만 사용 가능하므로 에러 

# loc, iloc 동일 -> 행이 먼저
print("df.iloc[:,2]:\n", df2.iloc[:,2]) #모든 행의 2번째 인덱스
#3, 7, 11, 15, 19
'''
df.iloc[:,2]:
 A     3
B     7
C    11
D    15
E    19
Name: 다, dtype: int64
'''
# print("df2[:,2]:\n", df2[:,2])
'''
TypeError: '(slice(None, None, None), 2)' is an invalid key
numpy에선 가능하나, pandas에서는 안된다
컬럼명이랑 인덱스 들어가야 한다
'''

#로우
print("df2.loc['A'] :\n", df2.loc['A'])
'''
df2.loc['A'] :
 가    1
나    2
다    3
라    4
Name: A, dtype: int64
'''
print("df2.loc[['A','C']]:\n", df2.loc[['A','C']])
'''
df2.loc['A','C']:
    가   나   다   라
A  1   2   3   4
C  9  10  11  12
'''

# print("df2.iloc['0'] :\n", df2.iloc['0'])
# TypeError: Cannot index by location index with a non-integer key
print("df2.iloc[[0,2]]:\n", df2.iloc[[0,2]])
'''
df2.iloc[[0,2]]:
    가   나   다   라
A  1   2   3   4
C  9  10  11  12
'''

#행렬
print("df2.loc[['A','B'],['나','다']]:\n",
        df2.loc[['A','B'],['나','다']])

'''
df2.loc[['A','B'],['나','다']]:
    나  다
A  2  3
B  6  7
'''

# 1개의 값만 확인
print("df2.loc['E', '다']:\n",
        df2.loc['E', '다'])
'''
df2.loc['E', '다']:
 19
'''

print("df2.iloc[4,2]:\n", df2.iloc[4,2])
'''
df2.iloc[4,2]:
 19
'''

print("df2.iloc[4][2] : \n", df2.iloc[4][2])
'''
df2.iloc[4][2] :
 19
'''