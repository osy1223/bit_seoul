# SFM에 save를 적용해서 제일 좋은 값만 남기고 다 지울것

import numpy as np
from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score

# 1. 데이터
x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size = 0.8,
    random_state = 77,
    shuffle=True
)

# 2. 모델
model = XGBRegressor(n_estimators=1000, learnin_rate=0.1)

# 3. 훈련
model.fit(x_train, y_train,
    verbose=1,
    eval_metric='rmse',
    eval_set=[(x_train,y_train),(x_test,y_test)])

# 4. 평가, 예측
result = model.evals_result()

y_pred = model.predict(x_test)

r2 = r2_score(y_pred, y_test)
print('r2:', r2)

# pickle 이용 save
import pickle
pickle.dump(model, open('./save/xgb_save/boston.pickle.dat', 'wb'))
print('저장완료')

