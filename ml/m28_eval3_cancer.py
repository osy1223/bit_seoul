import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size = 0.8,
    random_state=77,
    shuffle=True
)

# 2. 모델
model = XGBClassifier(n_estimators=100, learning_rate=0.1)
# model = XGBRegressor(learning_rate=0.01)


# 3. 훈련
model.fit(x_train, y_train, 
    verbose=1, #다 보여 준다(0 / 1 , False / True)
    eval_metric = 'logloss', #keras의 metrics와 동일
    eval_set =[(x_train,y_train),(x_test, y_test)] 
    #평가는 x_test, y_test & 지표는 RMSE
    #출력은 n_estimators만큼
)

# eval_metrics의 대표 param : rmse, mae, logloss, error, auc


# 4. 평가, 예측
result = model.evals_result()
# print("eval's results : ", result)

acc = model.score(x_test, y_test)
print('acc:', acc)

y_pred = model.predict(x_test)
print('최종정답률:', accuracy_score(y_test, y_pred))

'''
eval_metric = 'rmse',
acc: 0.9473684210526315
최종정답률: 0.9473684210526315

eval_metric = 'mae'
acc: 0.9473684210526315
최종정답률: 0.9473684210526315

eval_metric = 'error'
acc: 0.9473684210526315
최종정답률: 0.9473684210526315

eval_metric = 'auc'
acc: 0.9473684210526315
최종정답률: 0.9473684210526315

eval_metric = 'logloss'
acc: 0.9473684210526315
최종정답률: 0.9473684210526315
'''