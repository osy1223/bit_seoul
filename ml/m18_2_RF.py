# dicisionTree 가 앙상블 되어있는게 RandomForest
# DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, train_size=0.8, random_state=42
)
print(x_train.shape) #(455, 30)

# model = DecisionTreeClassifier(max_depth=4)
model = RandomForestClassifier(max_depth=4)
# max_depth=4 : 4번 잘랐다

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

print(acc)
0.9473684210526315
# 0.9649122807017544

print(model.feature_importances_)
# feature_importances_ : 특성 임포턴스
'''
[0.03046014 0.01430575 0.05812314 0.07116485 0.00557951 0.01617405
 0.03501429 0.13508548 0.00424615 0.00429758 0.01005619 0.00257663
 0.009148   0.03892272 0.00303778 0.00230542 0.00279077 0.00402171
 0.00154033 0.00437634 0.09773736 0.01527099 0.12460459 0.11152403
 0.01568527 0.01033333 0.04758865 0.09803241 0.01559827 0.0103983 ]
'''
# 기준치 잡는 건 본인 몫


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
