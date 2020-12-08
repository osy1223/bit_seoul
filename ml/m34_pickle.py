# dicisionTree 가 앙상블 되어있는게 RandomForest 에서 부스터 

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import  GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
# cmd에서 pip install xgboost 로 설치
# 다되면 Successfully installed xgboost-1.2.1
import matplotlib.pyplot as plt
import numpy as np

cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, train_size=0.8, random_state=42
)
# print(x_train.shape) #(455, 30)

model = XGBClassifier(max_depth=4)
# max_depth=4 : 4번 잘랐다

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(acc)

#파이썬 제공하는 저장
import pickle
pickle.dump(model, open('./save/xgb_save/cancer.pickle.dat', 'wb'))
print('저장완료')
'''
0.9649122807017544
저장완료
'''

# 모델+가중치 
model2 = pickle.load(open('./save/xgb_save/cancer.pickle.dat', 'rb'))
print('불러왔다')
acc2 = model2.score(x_test, y_test)
print('acc2:', acc2)
'''
불러왔다
acc2: 0.9649122807017544
'''