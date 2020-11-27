from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRFRegressor, plot_importance
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler




parameters =[
    {'n_estimators':[100,200,300], 
    'learning_rate':[0.1,0.3, 0.001, 0.01],
    'max_depth':[4,5,6]
    },
    {'n_estimators':[90,100,110], 
    'learning_rate':[0.1,0.001,0.01],
    'max_depth':[4,5,6],
    'colsample_bytree':[0.6,0.9,1],
    },
    {'n_estimators':[90,110], 
    'learning_rate':[0.1,0.001,0.5],
    'max_depth':[4,5,6],
    'colsample_bytree':[0.6,0.9,1],
    'colsample_bylevel':[0.6,0.,0.9]
    }
]

n_jobs = -1
# 가장 중요한 건 learning_rate!
# 파일을 완성하시오!

datasets = load_boston()
x = datasets.data
y = datasets.target
print("x.shape:", x.shape)


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

# 2. 모델
model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=2)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가
acc = model.score(x_test, y_test)
print("acc:", acc)


print("최적의 매개변수:", model.best_estimator_)
print("최적의 파라미터:", model.best_params_)

'''
# Pipeline + XGBRFRegressor + RandomizedSearchCV

# RandomizedSearchCV

# n_estimators = 300

# default XGBRFRegressor()

'''