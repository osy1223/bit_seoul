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

for thresh in thresholds :
    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)

    selection_model = XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score=r2_score(y_test, y_predict)
    
    print('Thresh=%.3f, n=%d, R2=%.2f%%' %(thresh, select_x_train.shape[1], score*100.0))

'''
Thresh=0.001, n=13, R2=92.21%
Thresh=0.004, n=12, R2=92.16%
Thresh=0.012, n=11, R2=92.03%
Thresh=0.012, n=10, R2=92.19%
Thresh=0.014, n=9, R2=93.08%
Thresh=0.015, n=8, R2=92.37%
Thresh=0.018, n=7, R2=91.48%
Thresh=0.030, n=6, R2=92.71%
Thresh=0.042, n=5, R2=91.74%
Thresh=0.052, n=4, R2=92.11%
Thresh=0.069, n=3, R2=92.52%
Thresh=0.301, n=2, R2=69.41%
Thresh=0.428, n=1, R2=44.98%
'''

'''
SelectFromModel 사용하면, 
Thresh :feature_importances_ 
n : n_components
feature_importances_의 개수만큼 for문 돕니다!
'''