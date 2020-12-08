import numpy as np

# 데이터 범위 한정 지어줍니다 (이상치, 극단값)
def outliers(data_out):
    quartile_1, quartile_3 = np.percentile(data_out, [25, 75])
    # 25, 75는 절대 값 아닙니다.
    #데이터의 25%와 75%를 quartile_1과 quartile_3에 넣어준다
    print('1사분위 : ', quartile_1) # 3.25
    print('3사분위 : ', quartile_3) # 97.5
    # 전체 데이터 4등분
    iqr = quartile_3 - quartile_1 # 94.25
    lower_bound = quartile_1 - (iqr*1.5)
    upper_bound = quartile_3 + (iqr*1.5)
    return np.where((data_out>upper_bound)|(data_out<lower_bound))
    # 다 잘라내면 데이터 손실이 너무 많으므로, 위아래로 1.5배수는 유지시켜준다.

a = np.array([1,2,3,4,10000,6,7,5000,90,100])

b = outliers(a)
print('이상치의 위치 :', b)
'''
이상치의 위치 : None

이상치의 위치 : (array([4, 7], dtype=int64),)
4번째, 7번째 자리 : 10000과 5000이 이상치다 
'''

# 스칼라 형식의 데이터에서만 가능!