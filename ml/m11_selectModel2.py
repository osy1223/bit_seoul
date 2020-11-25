# 리그래서 모델들 추출

import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

boston = pd.read_csv('./data/csv/boston_house_prices.csv',
    header=0,
    index_col=0,
    sep=',')

boston = boston[1:]
print(boston)

x = boston.iloc[:, :-1]
y = boston.iloc[:, -1:]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=44
)

allAlgorithms = all_estimators(type_filter='regressor') #리그래서 모델들 추출

for(name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률:', r2_score(y_test, y_pred))
    except:
        pass

import sklearn
print(sklearn.__version__) 
#0.22.1 버전에 문제 있어서 출력이 안됨 -> 버전 낮춰야함

'''
[506 rows x 13 columns]
ARDRegression 의 정답률: 0.7413660842736123
AdaBoostRegressor 의 정답률: 0.8461151829839103
BaggingRegressor 의 정답률: 0.8828922159785115
BayesianRidge 의 정답률: 0.7397243134288032
CCA 의 정답률: 0.7145358120880195
DecisionTreeRegressor 의 정답률: 0.8088012349921045
DummyRegressor 의 정답률: -0.0007982049217318821
ElasticNet 의 정답률: 0.6952835513419808
ElasticNetCV 의 정답률: 0.6863712064842076
ExtraTreeRegressor 의 정답률: 0.7262135731425631
ExtraTreesRegressor 의 정답률: 0.8961477168863623
GammaRegressor 의 정답률: -0.0007982049217318821
GaussianProcessRegressor 의 정답률: -5.586473869478007
GradientBoostingRegressor 의 정답률: 0.8991551809515332
HistGradientBoostingRegressor 의 정답률: 0.8843141840898427
HuberRegressor 의 정답률: 0.765516461537028
KernelRidge 의 정답률: 0.7635967087108912
Lars 의 정답률: 0.7440140846099284
LarsCV 의 정답률: 0.7499770153318335
Lasso 의 정답률: 0.683233856987759
LassoCV 의 정답률: 0.7121285098074346
LassoLars 의 정답률: -0.0007982049217318821
LassoLarsCV 의 정답률: 0.7477692079348519
LassoLarsIC 의 정답률: 0.7447915470841701
LinearRegression 의 정답률: 0.7444253077310311
LinearSVR 의 정답률: 0.2634055377378125
MLPRegressor 의 정답률: 0.5245866859196375
MultiTaskElasticNet 의 정답률: 0.6952835513419808
MultiTaskElasticNetCV 의 정답률: 0.6863712064842078
MultiTaskLasso 의 정답률: 0.6832338569877592
MultiTaskLassoCV 의 정답률: 0.7121285098074348
NuSVR 의 정답률: 0.32492104048309933
OrthogonalMatchingPursuit 의 정답률: 0.5661769106723642
OrthogonalMatchingPursuitCV 의 정답률: 0.7377665753906504
PLSCanonical 의 정답률: -1.30051983252021
PLSRegression 의 정답률: 0.7600229995900804
PassiveAggressiveRegressor 의 정답률: -0.061373857711268576
PoissonRegressor 의 정답률: 0.7903831388798964
RANSACRegressor 의 정답률: 0.6934737632680064
RandomForestRegressor 의 정답률: 0.8876887781958678
Ridge 의 정답률: 0.746533704898842
RidgeCV 의 정답률: 0.7452747014482557
SGDRegressor 의 정답률: -2.4102100136699278e+25
SVR 의 정답률: 0.2867592174963418
TheilSenRegressor 의 정답률: 0.7738216901718866
TransformedTargetRegressor 의 정답률: 0.7444253077310311
TweedieRegressor 의 정답률: 0.6899090088434408
0.23.1
'''