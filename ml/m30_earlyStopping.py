#ML에도 evaluate 기능이 있지 않을까? score 말고!
#verbose & eval_metric & eval_set
#XGBoost에도 다 있으므로 하고 싶은 거 다 할 수 있다 

import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score

# 1. 데이터
x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size = 0.8,
    random_state=77,
    shuffle=True
)

# 2. 모델
model = XGBRegressor(n_estimators=10000, learning_rate=0.1)
# model = XGBRegressor(learning_rate=0.01)


# 3. 훈련
model.fit(x_train, y_train, 
    verbose=1, #다 보여 준다(0 / 1 , False / True)
    eval_metric =['rmse'], #keras의 metrics와 동일. RMSE를 쓰겠다. 
    eval_set =[(x_train,y_train),(x_test, y_test)],
    early_stopping_rounds=100 #keras의 earlyStopping 기능 
    #평가는 x_test, y_test & 지표는 RMSE # earlyStopping 20번 참겠다.
    #출력은 n_estimators만큼
)

# eval_metrics의 대표 param : rmse, mae, logloss, error, auc


# 4. 평가, 예측
result = model.evals_result()
# print("eval's results : ", result)

y_pred = model.predict(x_test)

r2 = r2_score(y_pred, y_test) #r2 싸이킷런에서 제공
print('r2 :', r2)
# r2 : 0.9039005810073445

'''
[9999]  validation_0-rmse:0.00147       validation_1-rmse:2.63822      
r2 : 0.9039005810073445

Stopping. Best iteration:
[33]    validation_0-rmse:1.51293       validation_1-rmse:2.56262
r2 : 0.9014827566359215
'''