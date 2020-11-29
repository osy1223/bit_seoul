
import pandas as pd
import numpy as np
from pandas import DataFrame as df

##############################################################이혼
divorce = pd.read_csv('./project3/divorces_2000-2015.csv', 
                        header=0, index_col=0, sep=',')

# divorce.info()
'''
Index: 4923 entries
Data columns (total 40 columns):
'''
# print(divorce.columns)
'''
Index(['Type_of_divorce', 'Nationality_partner_man', 'DOB_partner_man',
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

divorce = divorce[['Age_partner_man','Monthly_income_partner_man_peso',
                    'Age_partner_woman','Monthly_income_partner_woman_peso', 
                    'Level_of_education_partner_man', 'Level_of_education_partner_woman', 'Marriage_duration']]
print(divorce) # [4923 rows x 7 columns]

# 이혼일 기준으로 오름차순으로 변경
divorce = divorce.sort_values(['Divorce_date'], ascending=['True'])
print(divorce) 
print(divorce.dtypes)

'''
Date_of_marriage                      object / 이거때문에 
Level_of_education_partner_man        object non, 
OTRO다른, PREPARATORIA대학예비의, PRIMARIA기초적인, PROFESIONAL전문적인, SECUNDARIA중등의, SIN ESCOLARIDAD취학 기간
Employment_status_partner_man         object 
non, EMPLEADO 고용된, ESTABLECIMIENTO 창업, JORNALERO O PEON일급노동자, MIEMBRO DE COOPERATIVA 조합, NO TRABAJA무직, OBRERO 노동자, PATRON O EMPRESARIO 고용자, TRABAJA EN SU VIVIENDA 일한다,
TRABAJADOR NO REMUNERADO 무급노동자, TRABAJADOR POR SU PROPIA CUENTA O EN VIA PUBLICA 회계?
Level_of_education_partner_woman      object
Employment_status_partner_woman       object
'''

divorce.dropna(axis=0, how='any', inplace=True) 
print(divorce) # [2429 rows x 7 columns]


divorce['Level_of_education_partner_man'].replace({'OTRO':1,'PRIMARIA':2,'SECUNDARIA':3,  'PREPARATORIA':4, 'PROFESIONAL':5,'SIN ESCOLARIDAD':6}, inplace=True)
divorce['Level_of_education_partner_woman'].replace({'OTRO':1,'PRIMARIA':2,'SECUNDARIA':3,  'PREPARATORIA':4, 'PROFESIONAL':5,'SIN ESCOLARIDAD':6}, inplace=True)


print(divorce) #[2429 rows x 7 columns]
print(divorce.dtypes)

'''
Age_partner_man                      float64
Monthly_income_partner_man_peso      float64
Age_partner_woman                    float64
Monthly_income_partner_woman_peso    float64
Level_of_education_partner_man         int64
Level_of_education_partner_woman       int64
Marriage_duration                    float64
dtype: object
'''


#################################################################행복지수
happy = pd.read_csv('./project3/happiness.csv', 
                        header=0, index_col=0, sep=',')
# happy.info()
'''
Index: 147 entries, Afghanistan to Zimbabwe
Data columns (total 20 columns):
'''


# 결측치 바로 이전의 값으로 채우기
happy = happy.fillna(method='pad')
happy.info()
print(happy.shape) #(147, 20)





#################################################################은행
bank = pd.read_csv('./project3/BankChurners.csv', 
                        header=0, index_col=0, sep=',')
                        
# 데이터 확인
# bank.info()
'''
Int64Index: 10127 entries
Data columns (total 22 columns):
'''

bank = bank[['Customer_Age', 'Gender', 'Education_Level', 'Avg_Open_To_Buy',
            'Card_Category', 'Months_on_book', 'Credit_Limit', 'Total_Revolving_Bal',
            'Income_Category','Marital_Status']]

# print(bank) #[10127 rows x 10 columns]

# 나이 기준으로 오름차순으로 변경
bank = bank.sort_values(['Customer_Age'], ascending=['True'])
# print(bank.dtypes)
'''
Customer_Age             int64
Gender                  object  : M, F
Education_Level         object  : High School, College, Doctorate, Post-Graduate, Uneducated, Graduate, Unknown
Avg_Open_To_Buy        float64
Card_Category           object  : Blue, Gold, Platinum, Silver
Months_on_book           int64
Credit_Limit           float64
Total_Revolving_Bal      int64
Income_Category         object : $120K + //// $40K - $60K, //// $60K - $80K //// $80K - $120K  //// Less than $40K //// Unknown
Marital_Status          object : Divorced, Married, Single, Unknown
dtype: object
'''

bank.replace("Unknown", np.nan, inplace=True) 
bank.dropna(axis=0, how='any', inplace=True) 
print(bank)

bank['Gender'].replace({'M':1,'F':2}, inplace=True)
bank['Education_Level'].replace({'Uneducated':1,'High School':2, 'Graduate':3, 'College':4, 'Doctorate':5, 'Post-Graduate':6}, inplace=True)
bank['Card_Category'].replace({'Blue':1,'Silver':2, 'Gold':3, 'Platinum':4}, inplace=True)
bank['Income_Category'].replace({'Less than $40K':4000,'$40K - $60K':5000, '$60K - $80K':7000, '$80K - $120K':10000, '$120K +':12000}, inplace=True)
bank['Marital_Status'].replace({'Single':1,'Married':2, 'Divorced':3}, inplace=True)

print(bank)
print(bank.dtypes)




# #################################################################행복지수
# happy = pd.read_csv('./project3/happiness.csv', 
#                         header=0, index_col=0, sep=',')
# # happy.info()
# '''
# Index: 147 entries, Afghanistan to Zimbabwe
# Data columns (total 20 columns):
# '''


# # 결측치 바로 이전의 값으로 채우기
# happy = happy.fillna(method='pad')
# happy.info()
# print(happy.shape) #(147, 20)
