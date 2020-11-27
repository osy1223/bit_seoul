# 실습
# 1. 상단 모델에 그리드서치 또는 랜덤서치 적용
# 2. 최적의 R2값과 피쳐임포턴스 구할 것
# 3. 위 쓰레드값으로 ScaleFromModel을 구해서 최적의 피쳐 개수를 구할 것
#    위 피쳐 개수로 데이터(피쳐)를 수정(삭제)해서 그리드서치 또는 랜덤서치 적용해서 최적의 R2값
# 1번값과 2번값을 비교해볼 것

from xgboost import XGBRegressor
from sklearn.datasets import load_boston #회귀모델
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor


# 1. 데이터
x, y = load_boston(return_X_y=True)

# 1.1 데이터 전처리 (train_test_split)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=66)


----------------------------------------------------------

parameters = [
    {'n_estimators' : [100,200],
    'max_depth' : [6,8,10,12],
    'min_samples_leaf' : [3,5,7,9],
    'min_samples_split' : range(2,7),
    'n_jobs':[-1]},
]

kfold = KFold(n_splits=8, shuffle=True) #n_splits : 전체 데이터 중 몇개로 조각낼지
# model = SVC()
model = RandomizedSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=2) 

----------------------------------------------------------

model = XGBRegressor(n_jobs=-1)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('R2 :', score)

# feature_importances_ sort 함수!
thresholds = np.sort(model.feature_importances_) #디폴트 오름차순 정렬
print(thresholds)

for thresh in thresholds :
    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)

    selection_model = XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score=r2_score(y_test, y_predict)
    
    print('Thresh=%.3f, n=%d, R2=%.2f%%' %(thresh, select_x_train.shape[1], score*100.0))