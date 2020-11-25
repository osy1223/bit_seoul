import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

wine = pd.read_csv('./data/csv/winequality-white.csv',
    sep=';', header=0)

y = wine['quality']
x = wine.drop('quality', axis=1)
#quality를 뺀 나머지를 x로 주겠다 (axis=1)

print(x) #[4898 rows x 11 columns]
print(y) #Name: quality, Length: 4898, dtype: int64 (4898,)

newlist = []
for i in list(y):
    if i <=4:
        newlist +=[0]
    elif i <=7:
        newlist +=[1]
    else :
        newlist +=[2]
# newlist를 범위지정해서 y라벨링 0,1,2로 데이터 전처리

y = newlist

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7
)

# 머신 러닝 모델에 이어라 
# 3. 모델
model = RandomForestClassifier()

# 4. 훈련
model.fit(x_train, y_train)

# 5. 평가, 예측
score = model.score(x_test, y_test)
print('model.score :', score)

y_predict = model.predict(x_test)

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score:',acc_score)

'''
model.score : 0.9448979591836735
accuracy_score: 0.9448979591836735
'''