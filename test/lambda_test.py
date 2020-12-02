import numpy as np
import pandas as pd
from pandas import DataFrame

np.random.seed(123)
df = DataFrame(np.random.randn(10,5),
                columns=['a','b','c','d','e'])
df['group'] = ['gr1', 'gr1', 'gr1', 'gr1', 'gr1',
                'gr2', 'gr2', 'gr3', 'gr3', 'gr3']

df = df.set_index('group')

# print(df)

'''
              a         b         c         d         e
group
gr1   -1.085631  0.997345  0.282978 -1.506295 -0.578600
gr1    1.651437 -2.426679 -0.428913  1.265936 -0.866740
gr1   -0.678886 -0.094709  1.491390 -0.638902 -0.443982
gr1   -0.434351  2.205930  2.186786  1.004054  0.386186
gr1    0.737369  1.490732 -0.935834  1.175829 -1.253881
gr2   -0.637752  0.907105 -1.428681 -0.140069 -0.861755
gr2   -0.255619 -2.798589 -1.771533 -0.699877  0.927462
gr3   -0.173636  0.002846  0.688223 -0.879536  0.283627
gr3   -0.805367 -1.727669 -0.390900  0.573806  0.338589
gr3   -0.011830  2.392365  0.412912  0.978736  2.238143
'''

def plus_ten(x):
    return x+10
# print(plus_ten(1))
# print(lambda x : x+10)

# print((lambda x: x+10)(1)) #11

corr_with_d = lambda x: x.corrwith(x['e'])
grouped = df.groupby('group')
grouped.apply(corr_with_d)
print(corr_with_d)