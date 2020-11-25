import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC

'''
datasets = load_iris()
x = datasets.data
y = datasets.target
'''

# 1.데이터
x, y = load_iris(return_X_y=True) #데이터셋에 2가지 방법이 있습니다

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7
)

# # OneHotEncoding
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# x_predict = x_train[:10]
# y_real = y_train[:10]

# 2.모델링
# model = Sequential()
# model.add(Dense(500, input_shape=(4,)))
# model.add(Dense(20))
# model.add(Dense(3, activation='softmax'))

# model.summary()
model = LinearSVC()

# 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# model.fit(x_train, y_train, epochs=400, batch_size=32, verbose=2, validation_split=0.4)
model.fit(x_train, y_train)

# 평가, 예측
# loss, acc = model.evaluate(x_test, y_test, batch_size=32)
# print("loss : ",loss)
# print("acc : ",acc)

# y_predict = model.predict(x_predict)
# y_predict_recovery = np.argmax(y_predict, axis=1)
# y_real = np.argmax(y_real, axis=1)
# print('예측값 : ',y_predict_recovery)
# print('실제값 : ',y_real) 

result = model.score(x_test, y_test)
print('score:',result)
'''
score: 0.9333333333333333
score: 0.9777777777777777
'''

# y_predict = model.predict(x_test)

'''
loss :  0.17748503386974335
acc :  0.9333333373069763
예측값 :  [1 2 2 1 2 2 2 0 0 0]
실제값 :  [1 2 2 1 2 2 2 0 0 0]

loss :  0.023001350462436676
acc :  1.0
예측값 :  [0 1 1 1 2 2 2 0 2 1]
실제값 :  [0 1 1 1 2 2 2 0 2 1]
'''


'''
sklean(머신러닝) 사용하면 ?

1. 데이터 구성 
(원핫인코딩 안해도 됨)
2. 모델
다중분류 : LinearSVC()
3. 훈련
model.fit(x_train, y_train)
4. 평가, 예측
model.score(x_test, y_test)

'''