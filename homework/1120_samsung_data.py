import numpy as np
import pandas as pd
from pandas import DataFrame

# csv 파일 불러오기 (인코딩 주의)
samsung = pd.read_csv('./data/csv/삼성전자 1120.csv',
            header=0, index_col=0, sep=',', encoding='cp949')

print('samsung:',samsung)
'''
samsung:                    시가         고가         저가  ...        외국계        
프로그램    외인비
일자                                           ...
2020/11/20     63,900     65,000     63,900  ...   -482,304    -604,078  56.53       
2020/11/19     64,100     64,800     63,900  ...   -766,372  -1,013,537  56.53       
2020/11/18     65,700     66,200     64,700  ...   -943,301  -2,310,858  56.54       
2020/11/17     67,000     67,000     65,600  ...    445,333     685,845  56.56       
2020/11/16     64,000     66,700     63,900  ...  5,190,222     456,517  56.53       
...               ...        ...        ...  ...        ...         ...    ...       
2018/03/23  2,517,000  2,536,000  2,480,000  ...    -25,773     -34,920  52.36       
2018/03/22  2,553,000  2,589,000  2,552,000  ...      7,441      23,952  52.39       
2018/03/21  2,589,000  2,589,000  2,553,000  ...    -11,651       9,610  52.38       
2018/03/20  2,535,000  2,560,000  2,505,000  ...    -16,199      -5,055  52.39       
2018/03/19  2,531,000  2,567,000  2,522,000  ...    -15,934      -2,128  52.40       

[660 rows x 16 columns]
'''

# 일부 데이터만 가져오기
samsung = samsung[['시가','고가','저가','거래량','종가']]
print(samsung)
'''
                   시가         고가         저가         종가         거래량
일자
2020/11/20     63,900     65,000     63,900     65,000   4,389,086
2020/11/19     64,100     64,800     63,900     64,600  16,590,290
2020/11/18     65,700     66,200     64,700     64,800  22,963,790
2020/11/17     67,000     67,000     65,600     65,700  30,204,089
2020/11/16     64,000     66,700     63,900     66,300  36,354,334
...               ...        ...        ...        ...         ...
2018/03/23  2,517,000  2,536,000  2,480,000  2,486,000     297,099
2018/03/22  2,553,000  2,589,000  2,552,000  2,589,000     169,082
2018/03/21  2,589,000  2,589,000  2,553,000  2,553,000     178,104
2018/03/20  2,535,000  2,560,000  2,505,000  2,560,000     163,865
2018/03/19  2,531,000  2,567,000  2,522,000  2,537,000     164,377

[660 rows x 5 columns]
'''
# 일자 기준으로 오름차순으로 변경
samsung = samsung.sort_values(['일자'], ascending=['True'])
print(samsung)
'''
                   시가         고가         저가         종가         거래량
일자
2018/03/19  2,531,000  2,567,000  2,522,000  2,537,000     164,377
2018/03/20  2,535,000  2,560,000  2,505,000  2,560,000     163,865
2018/03/21  2,589,000  2,589,000  2,553,000  2,553,000     178,104
2018/03/23  2,517,000  2,536,000  2,480,000  2,486,000     297,099
...               ...        ...        ...        ...         ...
2020/11/16     64,000     66,700     63,900     66,300  36,354,334
2020/11/17     67,000     67,000     65,600     65,700  30,204,089
2020/11/18     65,700     66,200     64,700     64,800  22,963,790
2020/11/19     64,100     64,800     63,900     64,600  16,590,290
2020/11/20     63,900     65,000     63,900     65,000   4,389,086

'''

# 중간에 더미 데이터 있어서, 빼고 가져오기
samsung = samsung.iloc[39:659]
print(samsung)
'''
                시가      고가      저가      종가         거래량
일자
2018/05/15  50,200  50,400  49,100  49,200  18,709,146
2018/05/16  49,200  50,200  49,150  49,850  15,918,683
2018/05/17  50,300  50,500  49,400  49,400  10,365,440
2018/05/18  49,900  49,900  49,350  49,500   6,706,570
2018/05/21  49,650  50,200  49,100  50,000   9,020,998
...            ...     ...     ...     ...         ...
2020/11/13  61,300  63,200  61,000  63,200  31,508,829
2020/11/16  64,000  66,700  63,900  66,300  36,354,334
2020/11/17  67,000  67,000  65,600  65,700  30,204,089
2020/11/18  65,700  66,200  64,700  64,800  22,963,790
2020/11/19  64,100  64,800  63,900  64,600  16,590,290

[620 rows x 5 columns]
'''

#콤마 제거 후 문자를 정수로 변환
for i in range(len(samsung.index)):
    for j in range(len(samsung.iloc[i])):
        samsung.iloc[i,j] = int(samsung.iloc[i,j].replace(',',''))
print(samsung)
'''
               시가     고가     저가       거래량     종가
일자
2018/05/15  50200  50400  49100  18709146  49200
2018/05/16  49200  50200  49150  15918683  49850
2018/05/17  50300  50500  49400  10365440  49400
2018/05/18  49900  49900  49350   6706570  49500
2018/05/21  49650  50200  49100   9020998  50000
...           ...    ...    ...       ...    ...
2020/11/16  64000  66700  63900  36354334  66300
2020/11/17  67000  67000  65600  30204089  65700
2020/11/18  65700  66200  64700  22963790  64800
2020/11/19  64100  64800  63900  16590290  64600

[620 rows x 5 columns]
'''

# numpy 파일로 변경 후 저장

samsung = samsung.to_numpy()
print(type(samsung))
'''
<class 'numpy.ndarray'>
'''

np.save('./data/samsung.npy', arr=samsung)
print(samsung)
print(samsung.shape)
'''
(620, 5)
'''