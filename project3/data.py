import pandas as pd
import numpy as np
from pandas import DataFrame as df
import matplotlib.pyplot as plt

#################################################################은행
bank = pd.read_csv('./project3/BankChurners.csv', 
                        header=0, index_col=0, sep=',')
                        
# 데이터 확인
# bank.info()
# print(bank)
# print('bank.columns', bank.columns)
'''
bank.columns Index(['Attrition_Flag', 'Customer_Age', 'Gender', 'Dependent_count',
       'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category',
       'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon',
       'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
       'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
       'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
       'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
       'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'],
      dtype='object')
'''
'''
Int64Index([768805383, 818770008, 713982108, 769911858, 709106358, 713061558,
            810347208, 818906208, 710930508, 719661558,
            ...
            712503408, 713755458, 716893683, 710841183, 713899383, 772366833,
            710638233, 716506083, 717406983, 714337233],
           dtype='int64', name='CLIENTNUM', length=10127)
'''
'''
Int64Index: 10127 entries
Data columns (total 22 columns):
'''

# 원하는 컬럼만 추출
bank = bank[['Customer_Age', 
              'Education_Level', 
              'Avg_Open_To_Buy',
              'Card_Category', 
              'Months_on_book', 
              'Credit_Limit', 
              'Total_Revolving_Bal',
              'Income_Category',
              'Marital_Status']]

# bank.info()
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
# bank.info()
##############################################################이혼
divorce = pd.read_csv('./project3/divorces_2000-2015.csv', 
                        header=0, index_col=0, sep=',')

# 데이터 확인
# divorce.info()
# print('divorce.columns', divorce.columns)
'''
divorce.columns Index(['Type_of_divorce', 'Nationality_partner_man', 'DOB_partner_man',
       'Place_of_birth_partner_man', 'Birth_municipality_of_partner_man',
       'Birth_federal_partner_man', 'Birth_country_partner_man',
       'Age_partner_man', 'Residence_municipality_partner_man',
       'Residence_federal_partner_man', 'Residence_country_partner_man',
       'Monthly_income_partner_man_peso', 'Occupation_partner_man',
       'Place_of_residence_partner_man', 'Nationality_partner_woman',
       'DOB_partner_woman', 'DOB_registration_date_partner_woman',
       'Place_of_birth_partner_woman', 'Birth_municipality_of_partner_woman',
       'Birth_federal_partner_woman', 'Birth_country_partner_woman',
       'Age_partner_woman', 'Place_of_residence_partner_woman',
       'Residence_municipality_partner_woman',
       'Residence_federal_partner_woman', 'Residence_country_partner_woman',
       'Occupation_partner_woman', 'Monthly_income_partner_woman_peso',
       'Date_of_marriage', 'Marriage_certificate_place',
       'Marriage_certificate_municipality', 'Marriage_certificate_federal',
       'Level_of_education_partner_man', 'Employment_status_partner_man',
       'Level_of_education_partner_woman', 'Employment_status_partner_woman',
       'Marriage_duration', 'Marriage_duration_months', 'Num_Children',
       'Custody'],
      dtype='object')
'''

'''
Index: 4923 entries
Data columns (total 40 columns):
'''
# print(divorce.columns)

# 원하는 컬럼만 추출
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
# print(divorce) 
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
age2 = divorce['Age_partner_woman']
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
# result.info()
# divorce.info()
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

# np.save('./project3/merge_x.npy', arr=merge_x)
# np.save('./project3/merge_y.npy', arr=merge_y)
# np.save('./project3/merge_index.npy', arr=merge_index)

# print(merge_x)
# print(merge_x.shape) 

# print(merge_y)
# print(merge_y.shape) 

merge[:15].plot.bar(rot=0)
plt.title("Income Divorce")
plt.xlabel('feature')
plt.ylabel('divorce')
plt.show()
print('merge[1]',merge[1])