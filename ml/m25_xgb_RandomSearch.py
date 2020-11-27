# 과적합 방지
# 1. 훈련데이터량을 늘린다
# 2. 피쳐수를 줄인다
# 3. reguraization

parameters = [
    {"n_estimators":[100,200,300],"learning_rate":[0.1,0.3,0.001,0.01],
    "max_depth":[4,5,6]},
    {"n_estimators":[90,100,110],"learning_rate":[0.1,0.001,0.01],
    "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1]},
    {"n_estimators":[100,200,300],"learning_rate":[0.1,0.09,1],
    "colsample_bylevel":[0.6,0.7,0.8]}
]


from sklearn.datasets import load_boston
datasets = load_boston()
x = datasets.data
y = datasets.target
print("x.shape:", x.shape)


from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test = train_test_split(
    x,y, random_state=44, shuffle=True, test_size=0.2)



from xgboost import XGBClassifier, XGBRFRegressor, plot_importance
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score

kfold = KFold(n_splits=5, shuffle=True)


# model = XGBRFRegressor(max_depth=max_depth, learning_rate=learning_rate,
#                         n_estimators=n_estimators, n_jobs=n_jobs,
#                         colsample_bylevel = colsample_bylevel,
#                         colsample_bytree=colsample_bytree )
model = RandomizedSearchCV(XGBRFRegressor(), 
                    parameters, 
                    cv=kfold, 
                    verbose=2) # kfold가 5번 x 20번 = 총 100번

# score 디폴트로 했던 놈과 성능 비교



model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print("acc:", acc)


print("최적의 매개변수:", model.best_estimator_)
print("최적의 파라미터:", model.best_params_)

y_predict = model.predict(x_test)
print('최종정답률:',r2_score(y_test, y_predict))

'''
# RandomizedSearchCV

# n_estimators = 300

# default XGBRFRegressor()
'''