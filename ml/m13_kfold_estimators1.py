# 분류

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
iris = pd.read_csv('./data/csv/iris_ys.csv',
    header=0,
    index_col=0)

x = iris.iloc[:, :4]
y = iris.iloc[:, 4]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=44
)

kfold = KFold(n_splits=5, shuffle=True) #n_splits : 전체 데이터 중 몇개로 조각낼지

allAlgorithms = all_estimators(type_filter='classifier') #클래스파이어 모델들을 

for(name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        # model.fit(x_train, y_train, cv=kfold)
        # y_pred = model.predict(x_test)
        print(name, '의 정답률:')
        print(scores)
    except:
        continue

'''
AdaBoostClassifier 의 정답률:
[0.95833333 0.95833333 0.91666667 0.875      0.91666667]
BaggingClassifier 의 정답률:
[0.91666667 1.         1.         0.95833333 0.91666667]
BernoulliNB 의 정답률:
[0.20833333 0.25       0.33333333 0.25       0.20833333]
CalibratedClassifierCV 의 정답률:
[0.91666667 0.875      0.875      0.875      0.91666667]
CategoricalNB 의 정답률:
[0.91666667 0.95833333 0.95833333 0.91666667 0.91666667]
CheckingClassifier 의 정답률:
[0. 0. 0. 0. 0.]
ComplementNB 의 정답률:
[0.625      0.54166667 0.70833333 0.625      0.79166667]
DecisionTreeClassifier 의 정답률:
[0.95833333 0.83333333 0.95833333 1.         0.875     ]
DummyClassifier 의 정답률:
[0.5        0.16666667 0.33333333 0.29166667 0.41666667]
ExtraTreeClassifier 의 정답률:
[0.91666667 0.79166667 0.875      0.875      0.91666667]
ExtraTreesClassifier 의 정답률:
[0.875      0.91666667 1.         1.         0.91666667]
GaussianNB 의 정답률:
[0.95833333 1.         0.91666667 0.95833333 1.        ]
GaussianProcessClassifier 의 정답률:
[0.95833333 1.         0.91666667 0.83333333 0.95833333]
GradientBoostingClassifier 의 정답률:
[0.91666667 0.91666667 1.         0.95833333 1.        ]
HistGradientBoostingClassifier 의 정답률:
[1.         0.91666667 0.91666667 0.95833333 0.91666667]
KNeighborsClassifier 의 정답률:
[1.         0.95833333 1.         0.875      0.95833333]
LabelPropagation 의 정답률:
[0.91666667 1.         0.95833333 0.95833333 0.95833333]
LabelSpreading 의 정답률:
[1.         0.95833333 0.91666667 0.95833333 0.83333333]
LinearDiscriminantAnalysis 의 정답률:
[1.         1.         0.91666667 0.875      1.        ]
LinearSVC 의 정답률:
[0.95833333 1.         0.91666667 0.91666667 1.        ]
LogisticRegression 의 정답률:
[0.95833333 1.         1.         0.83333333 0.95833333]
LogisticRegressionCV 의 정답률:
[1.         0.83333333 0.91666667 0.95833333 0.95833333]
MLPClassifier 의 정답률:
[1.         0.95833333 0.95833333 0.875      0.91666667]
MultinomialNB 의 정답률:
[0.58333333 0.625      0.75       0.95833333 1.        ]
NearestCentroid 의 정답률:
[1.         0.91666667 0.875      0.875      1.        ]
NuSVC 의 정답률:
[0.91666667 1.         0.91666667 0.91666667 1.        ]
PassiveAggressiveClassifier 의 정답률:
[0.875      0.79166667 0.75       0.95833333 0.70833333]
Perceptron 의 정답률:
[0.45833333 0.58333333 0.70833333 0.91666667 0.5       ]
QuadraticDiscriminantAnalysis 의 정답률:
[1.         0.95833333 0.875      1.         1.        ]
RadiusNeighborsClassifier 의 정답률:
[1.         0.875      0.875      0.95833333 1.        ]
RandomForestClassifier 의 정답률:
[0.95833333 0.875      1.         0.91666667 1.        ]
RidgeClassifier 의 정답률:
[0.875      0.91666667 0.83333333 0.875      0.83333333]
RidgeClassifierCV 의 정답률:
[0.91666667 0.79166667 0.83333333 0.83333333 0.83333333]
SGDClassifier 의 정답률:
[0.75       1.         0.625      0.83333333 0.79166667]
SVC 의 정답률:
[1.         1.         0.91666667 0.95833333 0.83333333]
'''