# gridSearch 전체를 다 찾아서 최적의 파라미터를 찾겠다.

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
# from sklearn.utils.testing import all_estimators

warnings.filterwarnings('ignore')

# 1. 데이터
iris = pd.read_csv('./data/csv/iris_ys.csv',
    header=0,
    index_col=0)

x = iris.iloc[:, :4]
y = iris.iloc[:, 4]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=66
)

# parameters = [
#     {"C":[1,10,100,1000], 'kernal1' :['linear']},
#     {"C":[1,10,100,1000], 'kernal1':['rbf'], 'gamma':[0.001, 0.0001]},
#     {"C":[1,10,100,1000], 'kernal1':['sigmoid'], 'gamma':[0.001, 0.0001]}
# ]

parameters = [{"C": [1, 10, 100, 1000], "kernel": ["linear"]}, # 1번: 1-linear 2번: 10-lenear, 3번: 100-leanear 4번: 100-leanear - 4번
              {"C": [1, 10, 100, 1000], "kernel": ["rbf"], "gamma":[0.001, 0.0001]}, # rbf 당 감마 4 * 2 = 8번
              {"C": [1, 10, 100, 1000], "kernel": ["sigmoid"], "gamma":[0.001, 0.0001]} # sigmoid 당 감마 4 * 2 = 8번
            ]
# 2. 모델
kfold = KFold(n_splits=5, shuffle=True)
# model = SVC()
model = GridSearchCV(SVC(), parameters, cv=kfold) #SVC라는 모델을 GridSearchCV로 쓰겠다

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
print('최적의 매개변수 :', model.best_estimator_)

y_predict = model.predict(x_test)
print('최종 정답률:', accuracy_score(y_test, y_predict))

'''
최적의 매개변수 : SVC(C=1, kernel='linear')
최종 정답률: 0.9666666666666667
'''