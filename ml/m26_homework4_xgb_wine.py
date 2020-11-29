# 와인은 csv 파일로 하기

# 과적합 방지
# 1. 훈련데이터량을 늘린다
# 2. 피쳐수를 줄인다
# 3. reguraization


parameters = [
    {"anyway__n_estimators":[100,200,300],"anyway__learning_rate":[0.1,0.3,0.001,0.01],
    "anyway__max_depth":[4,5,6]},
    {"anyway__n_estimators":[90,100,110],"anyway__learning_rate":[0.1,0.001,0.01],
    "anyway__max_depth":[4,5,6], "anyway__colsample_bytree":[0.6,0.9,1]},
    {"anyway__n_estimators":[100,200,300],"anyway__learning_rate":[0.1,0.09,1],
    "anyway__colsample_bylevel":[0.6,0.7,0.8]}
]

import pandas as pd
import numpy as np
wine = pd.read_csv('./data/csv/winequality-white.csv', 
                        header=0, # 첫 번 째 행 = 헤더다
                        index_col=0, # 컬럼 번호
                        encoding='CP949',
                        sep=';' # 구분 기호
                        )
y = wine['quality']
x = wine.drop('quality', axis=1).to_numpy()
print(x.shape)
print(y.shape)

print("========== feature importance cutting 시작 ==========")
from sklearn.model_selection import train_test_split 
from xgboost import XGBClassifier
x_train,x_test, y_train,y_test = train_test_split(
    x,y, random_state=44, shuffle=True, test_size=0.2)
model = XGBClassifier(max_depth=4)
model.fit(x_train, y_train)

import numpy as np
print("fi median:",np.median(model.feature_importances_))
print("fi mean:",np.mean(model.feature_importances_))

import numpy as np
def earseLowFI_index(fi_arr, low_value, input_arr):
    input_arr = input_arr.T
    temp = []
    for i in range(fi_arr.shape[0]):
        if fi_arr[i] >= low_value:
            temp.append(input_arr[i,:])
    temp = np.array(temp)
    temp = temp.T
    return temp

print("before f.i.cutting x.shape:",x.shape)
print("f.i:",model.feature_importances_)

x = earseLowFI_index(model.feature_importances_, np.mean(model.feature_importances_), x)
print("after f.i.cutting x.shape:",x.shape)
print("========== feature importance cutting 끝 ==========")


print("========== PCA를 위한 Scaler + PCA 시작 ==========")
# 1.3 scaler
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)


# 1.5 PCA
from sklearn.decomposition import PCA
cumsum_standard = 0.95
pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= cumsum_standard) +1
pca = PCA(n_components=d)
x = pca.fit_transform(x)
print("after pca x.shape", x.shape) # (70000, 202)
print("========== PCA를 위한 Scaler + PCA 끝 ==========")




print("========== train_test_split 시작 ==========")
# 1.2 train_test_split
from sklearn.model_selection import train_test_split 
x_train,x_test, y_train,y_test = train_test_split(
    x, y, test_size=0.2, random_state=44)
print("after x_train.shape",x_train.shape)
print("after x_test.shape",x_test.shape)
print("========== train_test_split 끝 ==========")



from xgboost import XGBClassifier, XGBRFRegressor, XGBClassifier, plot_importance
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

kfold = KFold(n_splits=5, shuffle=True)

pipe = Pipeline([('scaler', StandardScaler()),('anyway', XGBClassifier())])

# model = XGBRFRegressor(max_depth=max_depth, learning_rate=learning_rate,
#                         n_estimators=n_estimators, n_jobs=n_jobs,
#                         colsample_bylevel = colsample_bylevel,
#                         colsample_bytree=colsample_bytree )
# model = RandomizedSearchCV(XGBRFRegressor(), 
#                     parameters, 
#                     cv=kfold, 
#                     verbose=2) # kfold가 5번 x 20번 = 총 100번

model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=0)



model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print("acc:", acc)
print("최적의 매개변수:", model.best_estimator_)
print("최적의 파라미터:", model.best_params_)

y_predict = model.predict(x_test)
print('최종정답률:',accuracy_score(y_test, y_predict))



# acc: 0.6510204081632653
# 최적의 매개변수: Pipeline(steps=[('scaler', StandardScaler()),
#                 ('anyway',
#                  XGBClassifier(base_score=0.5, booster='gbtree',
#                                colsample_bylevel=0.8, colsample_bynode=1,
#                                colsample_bytree=1, gamma=0, gpu_id=-1,   
#                                importance_type='gain',
#                                interaction_constraints='', learning_rate=1,
#                                max_delta_step=0, max_depth=6,
#                                min_child_weight=1, missing=nan,
#                                monotone_constraints='()', n_estimators=200,
#                                n_jobs=0, num_parallel_tree=1,
#                                objective='multi:softprob', random_state=0,
#                                reg_alpha=0, reg_lambda=1, scale_pos_weight=None,
#                                subsample=1, tree_method='exact',
#                                validate_parameters=1, verbosity=None))]) 
# 최적의 파라미터: {'anyway__n_estimators': 200, 'anyway__learning_rate': 1, 'anyway__colsample_bylevel': 0.8}
# 최종정답률: 0.6510204081632653