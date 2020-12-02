import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.testing import all_estimators
from xgboost import XGBClassifier, XGBRegressor
from xgboost import plot_importance
import xgboost as xgb
import matplotlib as mpl


############################################ 데이터 불러오기
x = np.load('./project3/merge_x.npy')
y = np.load('./project3/merge_y.npy')
indexes = np.load('./project3/merge_index.npy', allow_pickle=True)

# print(indexes)
# print(indexes.shape) #(2,)
'''
[Index([       'Customer_Age',     'Education_Level',     'Avg_Open_To_Buy',
             'Card_Category',      'Months_on_book',        'Credit_Limit',
       'Total_Revolving_Bal',     'Income_Category',   'Marriage_duration',
                           2,                     1,                     0],
      dtype='object')
'''
# print(x.shape) #(18224, 12)
# print(y.shape) #(18224, 1)

########################################### train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=44, shuffle=True, train_size=0.8
)
# print(x_test.shape) #(3645, 12)
# print(y_test.shape) #(3645, 1)

############################################ y 라벨링
# y = to_categorical(y)

# parameters = [
#     {'n_estimators' : [100,200],
#     'max_depth' : [5],
#     'min_samples_leaf' : [3,5,7,9],
#     'min_samples_split' : range(2,7),
#     'n_jobs':[-1]},
# ]

########################################## 모델
# kfold = KFold(n_splits=5, shuffle=True) 
# model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=2) 
# RandomForestClassifier 모델을 GridSearchCV로 쓰겠다

# xgb_model = XGBClassifier(max_depth=6)

########################################## 모델
evals = [(x_test, y_test)]

xgb_model = XGBClassifier(
    n_estimators=200, 
    learning_rate=0.1, 
    max_depth=6)

# ############################################훈련
# model.fit(x_train, y_train)

# ############################################훈련
xgb_model.fit(x_train, y_train, 
    early_stopping_rounds=200, 
    eval_set=evals,
    verbose=True,
    eval_metric='merror' )

############################################ 피쳐임포턴스
# fig, ax = plt.subplots(figsize=(9,11))
# plot_importance(xgb_model, ax)
# print(ax)
# plt.show()


# def drawPlt(index, feature_importances):
#     n_features = len(xgb_model.feature_importances_)
#     plt.rcParams["figure.figsize"] = (15, 15)
#     plt.bar(np.arange(n_features), feature_importances)
#     plt.ylabel("Feature Importances")
#     plt.xlabel("Features")
#     plt.xticks(np.arange(n_features), index, rotation=90)

# drawPlt(indexes, xgb_model.feature_importances_)
# plt.show()


# ############################################평가, 예측
# print('최적의 매개변수 :', model.best_estimator_)
'''
최적의 매개변수 : RandomForestClassifier(max_depth=6, min_samples_leaf=3, n_estimators=200,
                       n_jobs=-1)

RandomForestClassifier 모델을 GridSearchCV로 사용시
최종 예측률 : 0.7706447187928669
'''

# print('최종 예측률:', accuracy_score(y_test,y_predict))
# y_predict = model.predict(x_test)
# print('최종 예측률:', accuracy_score(y_test,y_predict))
# y_predict_recovery = np.argmax(y_predict)
# y_real = np.argmax(y_test)

'''
RandomForestClassifier 모델을 GridSearchCV로 사용시
최종 예측률 : 0.7706447187928669

xgb 모델 사용시 
최종 예측률: 0.8093278463648834
real    :  [2. 2. 2. 2. 2. 2. 2. 1. 2. 1.]
predict :  [2. 2. 2. 2. 2. 2. 2. 3. 1. 1.]

real    :  [2. 2. 2. 2. 2. 2. 2. 1. 2. 1.]
predict :  [2. 2. 2. 2. 2. 2. 2. 1. 2. 1.]
최종 예측률: 0.8296296296296296
'''


############################################# 평가, 예측
y_predict = xgb_model.predict(x_test)
# print("real    : ", y_test[:10])
# print("predict : ",y_predict[:10])
print('최종 예측률:', accuracy_score(y_test, y_predict))

# 최종 예측률: 0.831275720164609
############################################ 시각화
# print(y_test.shape) # (3645,)  => 많으니까 100개만 출력 시도

fig = plt.figure( figsize = (12, 4) )
chart = fig.add_subplot(1,1,1)
chart.plot(y_test[:100], marker='o', color='blue', label='real value')
chart.plot(y_predict[:100], marker='^', color='red', label='predict value')
chart.set_title('real value vs predict value')
plt.xlabel('y_test')
plt.ylabel('y_predict')
plt.legend(loc = 'best') 
plt.show()

