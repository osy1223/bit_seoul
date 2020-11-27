# 과적합 방지
# 1. 훈련 데이터량을 늘린다
# 2. 피쳐수를 줄인다
# 3. reguraization '정규화, 일반화'

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRFRegressor, plot_importance
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler


# 1. 데이터

datasets = load_boston()
x = datasets.data
y = datasets.target
print("x.shape:", x.shape)


parameters = [
    {"anyway__n_estimators":[100,200,300],"anyway__learning_rate":[0.1,0.3,0.001,0.01],
    "anyway__max_depth":[4,5,6]},
    {"anyway__n_estimators":[90,100,110],"anyway__learning_rate":[0.1,0.001,0.01],
    "anyway__max_depth":[4,5,6], "anyway__colsample_bytree":[0.6,0.9,1]},
    {"anyway__n_estimators":[100,200,300],"anyway__learning_rate":[0.1,0.09,1],
    "anyway__colsample_bylevel":[0.6,0.7,0.8]}
]

# 1.1 데이터 전처리 (train_test_split)
x_train,x_test, y_train,y_test = train_test_split(
    x,y, random_state=44, shuffle=True, test_size=0.2)


kfold = KFold(n_splits=5, shuffle=True)

pipe = Pipeline([('scaler', StandardScaler()),('anyway', XGBRFRegressor())])

# model = XGBRFRegressor(max_depth=max_depth, learning_rate=learning_rate,
#                         n_estimators=n_estimators, n_jobs=n_jobs,
#                         colsample_bylevel = colsample_bylevel,
#                         colsample_bytree=colsample_bytree )
# model = RandomizedSearchCV(XGBRFRegressor(), 
#                     parameters, 
#                     cv=kfold, 
#                     verbose=2) # kfold가 5번 x 20번 = 총 100번

model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=2)
model.fit(x_train, y_train)

# 4. 평가, 예측
acc = model.score(x_test, y_test)
print("acc:", acc)


print("최적의 매개변수:", model.best_estimator_)
print("최적의 파라미터:", model.best_params_)

'''
Pipeline + XGBRFRegressor + RandomizedSearchCV
acc: -2.3755183488341207
최적의 매개변수: Pipeline(steps=[('scaler', StandardScaler()),
                ('anyway',
                 XGBRFRegressor(base_score=0.5, booster='gbtree',
                                colsample_bylevel=1, colsample_bytree=1,
                                gamma=0, gpu_id=-1, importance_type='gain',
                                interaction_constraints='', learning_rate=0.3,
                                max_delta_step=0, max_depth=4,
                                min_child_weight=1, missing=nan,
                                monotone_constraints='()', n_estimators=300,
                                n_jobs=0, num_parallel_tree=300,
                                objective='reg:squarederror', random_state=0,
                                reg_alpha=0, scale_pos_weight=1,
                                tree_method='exact', validate_parameters=1,
                                verbosity=None))])
최적의 파라미터: {'anyway__n_estimators': 300, 'anyway__max_depth': 4, 'anyway__learning_rate': 0.3}


# RandomizedSearchCV


# n_estimators = 300


# default XGBRFRegressor()
'''