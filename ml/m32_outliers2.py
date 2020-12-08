# 과제
# outliers를 행렬형태로 적용 할 수 있도록 수정

import numpy as np

# 데이터 범위 한정 지어줍니다 (이상치, 극단값)
def outliers(data_out):
    quartile_1, quartile_3 = np.percentile(data_out, [25, 75])
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

def remove_outlier_test(d_cp, column):
    fraud_column_date = d_cp[d_cp['Class']==0][column]
    quan25 = np.percentile(fraud_column_date.values, 25)
    quan75 = np.percentile(fraud_column_date.values, 75)

    iqr = quan75 - quan25
    iqr = iqr*1.5
    lowest = quan25 - iqr
    highest = quan75 + iqr
    outlier_index = fraud_column_date[(fraud_column_date<lowest)|(fraud_column_date>highest)].index
    print(len(outlier_index))
    d_cp.drop(outlier_index, axis=0, inplace=True)
    print(d_cp.shape)
    return d_cp

c = np.array([[1,2,3,4,10000,6,7,5000,90,100],
            10000,20000,3,40000,50000,60000,70000,8,90000,100000])
c = c.transpose()
print(c.shape)

r = remove_outlier_test(c)
print(r)