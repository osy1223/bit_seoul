import pandas as pd
import numpy as np
from pandas import DataFrame as df


##############################################################이혼
divorce = pd.read_csv('./project3/divorces_2000-2015_translated.csv', 
                        header=0, index_col=0, sep=',')

# divorce.info()
# print(divorce.columns)

divorce = divorce[['Age_partner_man',
                    'Age_partner_woman',
                    'Monthly_income_partner_man_peso',
                    'Monthly_income_partner_woman_peso', 
                    'Level_of_education_partner_man', 
                    'Level_of_education_partner_woman', 
                    'Marriage_duration']]

divorce_age = divorce[['Age_partner_man', 'Age_partner_woman']]
# print('divorce_age:',divorce_age) 

divorce_income = divorce.iloc[:, 2:4]
# print('divorce_income',divorce_income)

divorce_edu = divorce.iloc[:, 4:6]
# print('divorce_edu',divorce_edu)

# 이혼일 기준으로 오름차순으로 변경
divorce = divorce.sort_values(['Divorce_date'], ascending=['True'])
# print(divorce) /
# print(divorce.dtypes)

# '''
# Level_of_education_partner_man        
# object non, 
# OTRO다른, PREPARATORIA대학예비의, PRIMARIA기초적인, PROFESIONAL전문적인, SECUNDARIA중등의, SIN ESCOLARIDAD취학 기간
# Level_of_education_partner_woman      object
# '''

divorce.dropna(axis=0, how='any', inplace=True) 
# print(divorce) # [2429 rows x 7 columns]


divorce['Level_of_education_partner_man'].replace({'OTRO':1,'PRIMARIA':2,'SECUNDARIA':3,  'PREPARATORIA':4, 'PROFESIONAL':5,'SIN ESCOLARIDAD':6}, inplace=True)
divorce['Level_of_education_partner_woman'].replace({'OTRO':1,'PRIMARIA':2,'SECUNDARIA':3,  'PREPARATORIA':4, 'PROFESIONAL':5,'SIN ESCOLARIDAD':6}, inplace=True)


print(divorce.shape) #[2429 rows x 7 columns]
# print(divorce.dtypes)
