import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
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

# ########################################### OneHotEncoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_train.shape, y_test.shape)
# # (14579, 4) (3645, 4)

# ########################################### scaler
# # (함수는 2차원밖에 안되서 수동으로 줄인겁니다!)
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

########################################### reshape
x_train = x_train.reshape(14579,6,2)
x_test = x_test.reshape(3645,6,2)
print("reshape x:", x_train.shape, x_test.shape)
# reshape x: (14579, 6, 2) (3645, 6, 2)

############################################ y 라벨링
# y = to_categorical(y)

# parameters = [
#     {'n_estimators' : [100,200],
#     'max_depth' : [5],
#     'min_samples_leaf' : [3,5,7,9],
#     'min_samples_split' : range(2,7),
#     'n_jobs':[-1]},
# ]

########################################## 모델1
model = Sequential()
model.add(Conv1D(500, 2, input_shape=(6,2)))
model.add(Conv1D(200, 3))
model.add(Flatten())
model.add(Dense(200))
model.add(Dense(90))
model.add(Dense(80))
model.add(Dense(4, activation='softmax'))

model.summary()

########################################## 모델2
# kfold = KFold(n_splits=5, shuffle=True) 
# model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=2) 
# RandomForestClassifier 모델을 GridSearchCV로 쓰겠다

# xgb_model = XGBClassifier(max_depth=6)

########################################## 모델3
# evals = [(x_test, y_test)]

# xgb_model = XGBClassifier(
#     n_estimators=200, 
#     learning_rate=0.1, 
#     max_depth=6)

############################################# 컴파일
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['acc']
)

model.fit(x_train, y_train,
    epochs=100,
    batch_size=33,
    verbose=1,
    validation_split=0.5)

############################################# 평가, 예측
y_predict = model.predict(x_test)
print("real    : ", y_test[:10])
print("predict : ",y_predict[:10])

# RMSE, R2
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE :", RMSE(y_test,y_predict))

# R2 함수
from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)
print("R2 : ",r2)


# ############################################훈련
# xgb_model.fit(x_train, y_train, 
#     early_stopping_rounds=200, 
#     eval_set=evals,
#     verbose=True,
#     eval_metric='merror' )


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
# y_predict = model.predict(x_test)
# print("real    : ", y_test[:10])
# print("predict : ",y_predict[:10])
# print('최종 예측률:', accuracy_score(y_test, y_predict))

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

