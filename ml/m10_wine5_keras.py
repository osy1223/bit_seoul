import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

wine = pd.read_csv('./data/csv/winequality-white.csv',
    sep=';', header=0, index_col=0)

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
# y의 범위를 3~9 -> 0,1,2로 좁히는 것은
# 평가 방법을 바꾼 것이지 데이터 조작이 아니다


# OneHotEncoding
y = to_categorical(y)


# train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7, random_state=44
)

# scaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print('x_train.shape:', x_train.shape) #(3428, 10)
print('x_test.shape:', x_test.shape) #1470, 10)

# reshape
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1) 
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1) 
print('x_train.reshape:', x_train.shape) #(3428, 10, 1)
print('x_test.reshape:', x_test.shape) #(1470, 10, 1)

# 3. 모델
model = Sequential()
model.add(LSTM(1, input_shape=(10,1)))
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(500))
model.add(Dense(200))
model.add(Dense(3, activation='softmax'))

model.summary()

# 4. 훈련
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['acc']
)

model.fit(x_train, y_train,
    epochs=100,
    batch_size=5,
    verbose=1,
    validation_split=0.5 
)

# 5. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ",loss)
print("acc : ",acc)

y_predict = model.predict(x_test) # 평가 데이터 다시 넣어 예측값 만들기

# 원핫인코딩 후 원복
y_test = np.argmax(y_test, axis=1)
y_predict = np.array(y_predict)
print("y_test:\n", y_test)
print("y_predict:\n", y_predict)

'''
loss :  0.31622132658958435
acc :  0.9231292605400085
y_test:
 [1 1 1 ... 1 1 1]
y_predict:
 [[0.03695013 0.95310915 0.00994067]
 [0.0311842  0.9463246  0.02249116]
 [0.03435915 0.951425   0.01421583]
 ...
 [0.0238732  0.90824085 0.06788593]
 [0.02755463 0.93364614 0.03879931]
 [0.03555086 0.95241374 0.01203548]]
'''