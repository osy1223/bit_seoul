# dicisionTree 가 앙상블 되어있는게 RandomForest 에서 부스터 GradientBoostingClassifier
# DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import  GradientBoostingClassifier, GradientBoostingRegressor
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
# model = RandomForestClassifier(max_depth=4)
model = GradientBoostingClassifier(max_depth=4)
# max_depth=4 : 4번 잘랐다

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

print(acc)
# 0.9473684210526315
# 0.9649122807017544
# 0.956140350877193

print(model.feature_importances_)
# feature_importances_ : 특성 임포턴스
'''
[2.35942615e-05 3.66644283e-02 6.27557630e-04 1.00197673e-04
 1.20837308e-03 2.22866193e-04 4.58322225e-04 6.69964253e-01
 6.57120119e-04 5.49860302e-05 3.60367262e-03 1.89788919e-03
 3.02429679e-04 8.61426025e-03 2.05587020e-03 2.08505875e-03
 1.34973829e-02 1.55115362e-02 7.28350147e-04 5.20095285e-03
 5.65105990e-02 2.68763579e-02 5.59676950e-02 4.40516574e-03
 8.67007567e-03 2.19142053e-03 3.00289126e-03 7.84034099e-02
 4.02242138e-04 9.10416757e-05]
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