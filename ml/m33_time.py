from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston #회귀모델
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score


x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=66)

model = XGBRegressor(n_jobs=-1)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('R2 :', score)

thresholds = np.sort(model.feature_importances_) #디폴트 오름차순 정렬
print(thresholds)

import time

start1 = time.time()
for thresh in thresholds :
    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)

    selection_model = XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train, y_train, verbose=0)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score=r2_score(y_test, y_predict)
    
    # print('Thresh=%.3f, n=%d, R2=%.2f%%' %(thresh, select_x_train.shape[1], score*100.0))

start2 = time.time()
for thresh in thresholds :
    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)

    selection_model = XGBRegressor(n_jobs=11)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score=r2_score(y_test, y_predict)
    
    # print('Thresh=%.3f, n=%d, R2=%.2f%%' %(thresh, select_x_train.shape[1], score*100.0))
    # 보여주는 만큼 시간 delay 걸림

end = start2 - start1
print('그냥 걸린 시간 : ', end)
end2 = time.time() - start2
print('잡스 걸린 시간 : ', end2)

'''
n_jobs=6 (6코어 12쓰레드)
그냥 걸린 시간 :  2.6947929859161377
잡스 걸린 시간 :  1.4441375732421875

그냥 걸린 시간 :  2.6997792720794678
잡스 걸린 시간 :  1.2995259761810303
------------------------------------
n_jobs=8
그냥 걸린 시간 :  2.5980522632598877
잡스 걸린 시간 :  1.869999885559082
------------------------------------
n_jobs=11
그냥 걸린 시간 :  2.718729257583618
잡스 걸린 시간 :  2.468397855758667
------------------------------------
n_jobs=12
그냥 걸린 시간 :  2.685817003250122
잡스 걸린 시간 :  2.7087552547454834

그냥 걸린 시간 :  2.697786808013916
잡스 걸린 시간 :  2.7237160205841064

쓰레드는 안먹힘
여기서는 n_jobs=6이 제일 좋았음
'''