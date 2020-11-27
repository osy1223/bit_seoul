from sklearn.datasets import load_boston
from xgboost import XGBClassifier, XGBRFRegressor, plot_importance
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 다 한 사람은 모델을 완성해서 결과 주석으로 적어 놓을것

# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
print('x.shape:', x.shape)

# 1.1 데이터 전처리
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state=42
)

n_estimator = 300  #생성할 tree의 개수
learning_rate = 1  #학습 속도
colsample_bytree = 1  #각 트리마다의 feature 샘플링 비율, 보통 0.5 ~ 1 사용됨
colsample_bylevel = 1  #각 노드에서 사용되는 기능 (무작위로 선택됨)의 비율
max_deepth = 5  #트리의 최대 깊이, 보통 3-10 사이
n_jobs = -1

model = XGBRFRegressor(
    max_deepth=max_deepth,
    learning_rate=learning_rate,
    n_estimators=n_estimator,
    n_jobs=n_jobs,
    colsample_bylevel=colsample_bylevel,
    colsample_bytree=colsample_bytree
)

# score 디폴트로 했던 놈과 성능 비교

model.fit(x_train, y_train)

score = model.score(x_test, y_test)

print('score:', score)
# score: 0.8885450367658254
print(model.feature_importances_)
'''
[0.03438673 0.0107207  0.02308979 0.01400883 0.05360362 0.38742343
 0.01602715 0.07639714 0.01336571 0.03014113 0.05086887 0.01462501
 0.2753419 ]
'''

# 시각화
plot_importance(model)
plt.show()
