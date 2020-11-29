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

# 나이 기준으로 오름차순으로 변경
bank = bank.sort_values(['Customer_Age'], ascending=['True'])
# print(bank.dtypes)

bank.replace("Unknown", np.nan, inplace=True) 
bank.dropna(axis=0, how='any', inplace=True) 
# print(bank)

bank['Education_Level'].replace({'Uneducated':1,'High School':2, 'Graduate':3, 'College':4, 'Doctorate':5, 'Post-Graduate':6}, inplace=True)
bank['Card_Category'].replace({'Blue':1,'Silver':2, 'Gold':3, 'Platinum':4}, inplace=True)
bank['Income_Category'].replace({'Less than $40K':4000,'$40K - $60K':5000, '$60K - $80K':7000, '$80K - $120K':10000, '$120K +':12000}, inplace=True)
bank['Marital_Status'].replace({'Single':1,'Married':2, 'Divorced':3}, inplace=True)

print(bank.shape) #(7081, 9)
# print(bank.dtypes)



##############################################################이혼
divorce = pd.read_csv('./project3/divorces_2000-2015_translated.csv', 
                        header=0, index_col=0, sep=',')

# divorce.info()
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

divorce.dropna(axis=0, how='any', inplace=True) 
# print(divorce) # [2429 rows x 7 columns]


divorce['Level_of_education_partner_man'].replace({'OTRO':1,'PRIMARIA':2,'SECUNDARIA':3,  'PREPARATORIA':4, 'PROFESIONAL':5,'SIN ESCOLARIDAD':6}, inplace=True)
divorce['Level_of_education_partner_woman'].replace({'OTRO':1,'PRIMARIA':2,'SECUNDARIA':3,  'PREPARATORIA':4, 'PROFESIONAL':5,'SIN ESCOLARIDAD':6}, inplace=True)


print(divorce.shape) #[2429 rows x 7 columns]
# print(divorce.dtypes)

###########################################################데이터 합치기
merge = bank.append(divorce)
# print(merge)

merge = merge.fillna(merge.mean())
# print(merge)
print(merge.shape) #(9510, 17)
###########################################################train, test 나누기
# print(merge.columns)
'''
Index(['Customer_Age', 'Gender', 'Education_Level', 'Avg_Open_To_Buy',
       'Card_Category', 'Months_on_book', 'Credit_Limit',
       'Total_Revolving_Bal', 'Income_Category', 'Marital_Status',
       'Age_partner_man', 'Monthly_income_partner_man_peso',
       'Age_partner_woman', 'Monthly_income_partner_woman_peso',
       'Level_of_education_partner_man', 'Level_of_education_partner_woman',
       'Marriage_duration'],
'''
# print(bank.columns) 'Marital_Status':3 divorce
# print(divorce.columns) Marriage_duration 

# print(divorce['Marriage_duration'])

merge_x = merge[['Customer_Age', 'Gender', 'Education_Level', 'Avg_Open_To_Buy',
       'Card_Category', 'Months_on_book', 'Credit_Limit',
       'Total_Revolving_Bal', 'Income_Category', 'Age_partner_man', 'Monthly_income_partner_man_peso',
       'Age_partner_woman', 'Monthly_income_partner_woman_peso',
       'Level_of_education_partner_man', 'Level_of_education_partner_woman']]
print(merge_x) #[9510 rows x 15 columns]

merge_y = merge[['Marital_Status', 'Marriage_duration']]
print(merge_y) #[9510 rows x 2 columns]


###########################################################np.save
merge_x = merge_x.to_numpy()
print(type(merge_x)) #<class 'numpy.ndarray'>

merge_y = merge_y.to_numpy()
print(type(merge_y)) #<class 'numpy.ndarray'>

# np.save('./project3/merge_x.npy', arr=merge_x)
# np.save('./project3/merge_y.npy', arr=merge_y)

print(merge_x)
print(merge_x.shape) 

print(merge_y)
print(merge_y.shape) 