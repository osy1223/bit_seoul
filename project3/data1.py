import pandas as pd
import numpy as np
from pandas import DataFrame as df

#################################################################은행
bank = pd.read_csv('./project3/BankChurners.csv', 
                        header=0, index_col=0, sep=',')
                        
# 데이터 확인
# bank.info()
'''
Int64Index: 10127 entries
Data columns (total 22 columns):
'''

bank = bank[['Customer_Age', 
              'Education_Level', 
              'Avg_Open_To_Buy',
              'Card_Category', 
              'Months_on_book', 
              'Credit_Limit', 
              'Total_Revolving_Bal',
              'Income_Category',
              'Marital_Status']]

# print(bank) 
# aa = bank['Marital_Status'].groupby(bank['Marital_Status']).count()
# print(aa)
# 나이 기준으로 오름차순으로 변경
bank = bank.sort_values(['Customer_Age'], ascending=['True'])
# print(bank.dtypes)


# 데이터 정리
bank.replace("Unknown", np.nan, inplace=True) 
bank.dropna(axis=0, how='any', inplace=True) 
# print(bank)

bank['Education_Level'].replace({'Uneducated':1,'High School':2, 'Graduate':3, 'College':4, 'Doctorate':5, 'Post-Graduate':6}, inplace=True)
bank['Card_Category'].replace({'Blue':1,'Silver':2, 'Gold':3, 'Platinum':4}, inplace=True)
bank['Income_Category'].replace({'Less than $40K':4000,'$40K - $60K':5000, '$60K - $80K':7000, '$80K - $120K':10000, '$120K +':12000}, inplace=True)
bank['Marital_Status'].replace({'Single':1,'Married':2, 'Divorced':3}, inplace=True)
aa = bank['Marital_Status'].groupby(bank['Marital_Status']).count()
# print(aa)
# print(bank.shape) #(7081, 9)
# print(bank.dtypes)

##############################################################이혼
divorce = pd.read_csv('./project3/divorces_2000-2015.csv', 
                        header=0, index_col=0, sep=',')

# 데이터 확인
divorce.info()
'''
Index: 4923 entries
Data columns (total 40 columns):
'''
# print(divorce.columns)

divorce = divorce[['Age_partner_man',
                    'Age_partner_woman',
                    'Monthly_income_partner_man_peso',
                    'Monthly_income_partner_woman_peso', 
                    'Level_of_education_partner_man', 
                    'Level_of_education_partner_woman', 
                    'Marriage_duration']]
# print(divorce) # [4923 rows x 7 columns]

# 이혼일 기준으로 오름차순으로 변경
divorce = divorce.sort_values(['Divorce_date'], ascending=['True'])
# print(divorce) /
# print(divorce.dtypes)

'''
Level_of_education_partner_man        
object non, 
OTRO다른, PREPARATORIA대학예비의, PRIMARIA기초적인, PROFESIONAL전문적인, SECUNDARIA중등의, SIN ESCOLARIDAD취학 기간
Level_of_education_partner_woman      object
'''
divorce['Level_of_education_partner_man'].replace({'OTRO':1,'PRIMARIA':2,'SECUNDARIA':3,  'PREPARATORIA':4, 'PROFESIONAL':5,'SIN ESCOLARIDAD':6}, inplace=True)
divorce['Level_of_education_partner_woman'].replace({'OTRO':1,'PRIMARIA':2,'SECUNDARIA':3,  'PREPARATORIA':4, 'PROFESIONAL':5,'SIN ESCOLARIDAD':6}, inplace=True)
# print(divorce.shape) #[2429 rows x 7 columns]
# print(divorce.dtypes)

# 나이 데이터 합치기 
age1= divorce['Age_partner_man']
# print('age1',age1)
age2 = divorce['Age_partner_woman']
# print('age2',age2)
age = pd.concat([age1, age2], ignore_index=True)
# print('age:',age)
# print(age.shape) # (9846,)

# 학력 데이터 합치기
education1 = divorce['Level_of_education_partner_man']
education2 = divorce['Level_of_education_partner_woman']
education = pd.concat([education1, education2], ignore_index=True)
# print('education:',education)
# print(education.shape) #(9846,)


# 수입 데이터 합치기
income1 = divorce['Monthly_income_partner_man_peso']
income2 = divorce['Monthly_income_partner_woman_peso']
income = pd.concat([income1, income2], ignore_index=True)
income = income.sort_values()
# print('income3:',income)
# print(income.shape) # (9846,)


divorce = divorce[['Marriage_duration']]
# print(divorce) # [4923 rows x 1 columns]
# print(divorce.max()) #Marriage_duration    61.0
# print(divorce.min()) #Marriage_duration    1.0

result = pd.concat([age, education, income], axis=1)
result.dropna(axis=0, how='any', inplace=True) 
divorce = divorce.append(result)
divorce.fillna(divorce.mean(), inplace=True) 
# print(divorce) #[11143 rows x 4 columns]


##########################################################데이터 합치기
merge = bank.append(divorce)
print(merge) #[18224 rows x 13 columns]
aa = merge['Marital_Status'].groupby(merge['Marital_Status']).count()
# print(aa)

merge = merge.fillna(method='pad')
print(merge) #[18224 rows x 13 columns]
aa = merge['Marital_Status'].groupby(merge['Marital_Status']).count()
print('aa:',aa)

###########################################################x, y 나누기
# print('merge.columns',merge.columns)
'''
merge.columns Index([       'Customer_Age',     'Education_Level',     'Avg_Open_To_Buy',
             'Card_Category',      'Months_on_book',        'Credit_Limit',
       'Total_Revolving_Bal',     'Income_Category',      'Marital_Status',
                           2],
      dtype='object')
'''
# print(divorce['Marriage_duration'])
merge_x = merge[['Customer_Age', 'Education_Level', 'Avg_Open_To_Buy',
       'Card_Category', 'Months_on_book', 'Credit_Limit',
       'Total_Revolving_Bal', 'Income_Category','Marriage_duration', 2 ,1, 0]]
# print('merge_x:',merge_x)  # [18224 rows x 12 columns]
# 나이 : 0, 학벌 : 1, 수입: 2

merge_y = merge[['Marital_Status']]
# print('merge_y:',merge_y)  # [18224 rows x 1 columns]

# print(merge_x.head())
# print(merge_y.head())

merge_index = merge_x.columns, merge_y.columns
# print(merge_index)

# print(merge_index)
# print(merge_index.dtypes) 


# # ###########################################################np.save
merge_x = merge_x.to_numpy()
print(type(merge_x)) #<class 'numpy.ndarray'>

merge_y = merge_y.to_numpy()
print(type(merge_y)) #<class 'numpy.ndarray'>

np.save('./project3/merge_x.npy', arr=merge_x)
np.save('./project3/merge_y.npy', arr=merge_y)
np.save('./project3/merge_index.npy', arr=merge_index)

print(merge_x)
print(merge_x.shape) 

print(merge_y)
print(merge_y.shape) 