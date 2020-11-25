# 회귀

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import all_estimators

warnings.filterwarnings('ignore')

# 1. 데이터
boston = pd.read_csv('./data/csv/boston_house_prices.csv',
    header=0, index_col=0, sep=',')

print(boston)

boston = boston[1:]
# 이거 안 넣으면 에러 (데이터 확인해보면 압니당!!)

x = boston.iloc[:, :-1]
y = boston.iloc[:, -1:]
print(x.shape, y.shape) 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=44
)

# 2. 모델
allAlgorithms = all_estimators(type_filter='regressor') #리그래서 모델들 

kfold = KFold(n_splits=5, shuffle=True) #n_splits : 전체 데이터 중 몇개로 조각낼지

for(name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        
        print(name, '의 정답률:')
        print(scores)
    except:
        continue

'''
ARDRegression 의 정답률:
[0.68838963 0.69950803 0.55284136 0.73000504 0.79229478]
AdaBoostRegressor 의 정답률:
[0.84735787 0.76428195 0.84150383 0.84349249 0.67412109]
BaggingRegressor 의 정답률:
[0.75289068 0.78799263 0.76752659 0.83054541 0.88423655]
BayesianRidge 의 정답률:
[0.71208002 0.72746701 0.7834116  0.58377962 0.6686283 ]
CCA 의 정답률:
[0.63977744 0.77496711 0.75118066 0.50187629 0.38318092]
DecisionTreeRegressor 의 정답률:
[0.6570683  0.71322096 0.73907101 0.6522717  0.7465881 ]
DummyRegressor 의 정답률:
[-0.03065517 -0.00452891 -0.01684456 -0.00179345 -0.00010829]
ElasticNet 의 정답률:
[0.37778541 0.65579314 0.55980955 0.78771155 0.71047195]
ElasticNetCV 의 정답률:
[0.65887839 0.62575252 0.5938657  0.64898318 0.68151885]
ExtraTreeRegressor 의 정답률:
[0.61071754 0.90140466 0.63369228 0.49328543 0.80994146]
ExtraTreesRegressor 의 정답률:
[0.91427839 0.90014874 0.91279083 0.8498102  0.76562011]
GaussianProcessRegressor 의 정답률:
[-5.99366073 -6.79814701 -5.50067736 -5.98967768 -5.93263347]
GradientBoostingRegressor 의 정답률:
[0.90916901 0.82641061 0.93290362 0.77492978 0.7484227 ]
HistGradientBoostingRegressor 의 정답률:
[0.89342684 0.84411374 0.90952261 0.77345416 0.82906923]
HuberRegressor 의 정답률:
[0.68058895 0.77396166 0.65500002 0.7355917  0.3556861 ]
IsotonicRegression 의 정답률:
[nan nan nan nan nan]
KernelRidge 의 정답률:
[0.78030591 0.54504284 0.64425005 0.64216699 0.76153539]
Lars 의 정답률:
[0.73126217 0.67420495 0.74609098 0.59535457 0.69969196]
LarsCV 의 정답률:
[0.71614895 0.69826191 0.76640592 0.78789636 0.52487078]
Lasso 의 정답률:
[0.75817136 0.75392183 0.63230878 0.54809368 0.55763912]
LassoCV 의 정답률:
[0.66419317 0.71664828 0.62329097 0.7085075  0.6735394 ]
LassoLars 의 정답률:
[-0.00358635 -0.00632387 -0.10544139 -0.00845224 -0.02814675]
LassoLarsCV 의 정답률:
[0.77441828 0.38358811 0.69208554 0.7044637  0.79834066]
LassoLarsIC 의 정답률:
[0.8005172  0.60037729 0.72913903 0.65770158 0.67769895]
LinearRegression 의 정답률:
[0.67178005 0.73806997 0.75016534 0.61075173 0.72767842]
LinearSVR 의 정답률:
[ 0.61399849 -0.22578679  0.16187843  0.61604028  0.6133518 ]
MLPRegressor 의 정답률:
[ 0.63708641 -0.59797902  0.41750214  0.64018977  0.52304342]
MultiTaskElasticNet 의 정답률:
[0.70305116 0.60351388 0.59610797 0.74813822 0.5868519 ]
MultiTaskElasticNetCV 의 정답률:
[0.67412232 0.47210862 0.59302598 0.72343236 0.63541505]
MultiTaskLasso 의 정답률:
[0.57993132 0.66695286 0.62344367 0.59532277 0.66691795]
MultiTaskLassoCV 의 정답률:
[0.66900851 0.61447187 0.66114787 0.62690814 0.68611287]
NuSVR 의 정답률:
[-0.00782936  0.14265277  0.24861179  0.29346328  0.34613055]
OrthogonalMatchingPursuit 의 정답률:
[0.46883038 0.62810446 0.41201426 0.57303158 0.58707171]
OrthogonalMatchingPursuitCV 의 정답률:
[0.66900181 0.62212438 0.6817845  0.59759561 0.70126279]
PLSCanonical 의 정답률:
[-1.45754743 -2.82364823 -2.36054731 -2.37229329 -1.01669873]
PLSRegression 의 정답률:
[0.64984597 0.7939792  0.53067949 0.63888857 0.70881663]
PassiveAggressiveRegressor 의 정답률:
[-0.55935138  0.3391062   0.10672888  0.15056536  0.24797202]
RANSACRegressor 의 정답률:
[0.55475969 0.32280623 0.40364976 0.41478133 0.70482732]
RandomForestRegressor 의 정답률:
[0.86273546 0.88935941 0.87804927 0.80467992 0.89946905]
Ridge 의 정답률:
[0.71458704 0.59398122 0.67771706 0.65229664 0.80567777]
RidgeCV 의 정답률:
[0.74496589 0.6751401  0.67609371 0.63432503 0.75914101]
SGDRegressor 의 정답률:
[-2.83472093e+26 -1.08683282e+26 -1.67039916e+26 -6.29794223e+25
 -1.64692581e+26]
SVR 의 정답률:
[ 0.11533722  0.30962651  0.27312788 -0.10818449  0.3043599 ]
TheilSenRegressor 의 정답률:
[0.66575841 0.63520493 0.44811263 0.75487828 0.73912435]
TransformedTargetRegressor 의 정답률:
[0.75568572 0.67509682 0.62509265 0.74554265 0.60429386]
_SigmoidCalibration 의 정답률:
[nan nan nan nan nan]
'''