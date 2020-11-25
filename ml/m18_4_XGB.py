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
print(x_train.shape) #(455, 30)

# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier(max_depth=4)
# model = GradientBoostingClassifier(max_depth=4)
model = XGBClassifier(max_depth=4)
# max_depth=4 : 4번 잘랐다

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

print(acc)
# 0.9473684210526315
# 0.9649122807017544
# 0.956140350877193
# 0.9649122807017544

print(model.feature_importances_)
# feature_importances_ : 특성 임포턴스
'''
[6.5209707e-03 2.4647316e-02 5.1710028e-03 8.5555250e-03 3.9974600e-03
 4.6066064e-03 2.6123745e-03 4.4031221e-01 3.4104299e-04 2.0658956e-03
 1.2474300e-02 6.8988632e-03 1.7291201e-02 5.6377212e-03 3.1236352e-03
 3.2256048e-03 2.7834374e-02 6.4999197e-04 7.1520107e-03 0.0000000e+00
 7.2251976e-02 1.7675869e-02 8.9819685e-02 2.0090658e-02 1.0589287e-02
 0.0000000e+00 1.2581742e-02 1.8795149e-01 5.9212563e-03 0.0000000e+00]
'''

# 시각화
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(model)
plt.show()
