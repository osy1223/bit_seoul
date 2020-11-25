from sklearn.datasets import load_boston
from xgboost import XGBClassifier, XGBRFRegressor, plot_importance
from sklearn.model_selection import train_test_split


# 다 한 사람은 모델을 완성해서 결과 주석으로 적어 놓을것

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state=42
)

n_estimator = 300
learning_rate = 1
colsample_bytree = 1
colsample_bylevel = 1

max_deepth = 5
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
score = model.score(x_test, y_test)
print('점수 : ', socre)

print(model.feature_importances_)

plot_importance(model)
plt.show()

