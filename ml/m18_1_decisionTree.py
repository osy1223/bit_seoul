# DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, train_size=0.8, random_state=42
)
print(x_train.shape) #(455, 30)

model = DecisionTreeClassifier(max_depth=4)

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

print(acc)
# 0.9473684210526315

print(model.feature_importances_)
# feature_importances_ : 특성 임포턴스
'''
[0.         0.         0.         0.         0.         0.
 0.         0.70458252 0.         0.         0.01221069 0.
 0.         0.         0.         0.         0.00639525 0.0162341
 0.         0.0189077  0.05329492 0.05959094 0.05247428 0.
 0.00940897 0.         0.         0.06690062 0.         0.        ]
 5행 6열 = 30 // 총합은 1
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
