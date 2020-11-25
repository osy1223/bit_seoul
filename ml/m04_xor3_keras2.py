# 히든 레이어를 늘려서 평가 1 나오게

from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x_data = [[0,0], [1,0], [0,1], [1,1]]
y_data = [0,1,1,1]

# 2. 모델
# model = SVC()
model = Sequential()
model.add(Dense(1, input_dim=2, activation='relu'))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1, activation='sigmoid'))


# 3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)

# 4. 평가, 예측
y_predict = model.predict(x_data)
print(x_data,'의 예측결과:',y_predict)

# acc1 = model.score(x_data, y_predict)
# print('model.score:', acc1)

acc1 = model.evaluate(x_data, y_data)
print('model.evaluate:', acc1)

'''
[[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측결과: [[0.06711014]
 [0.99861825]
 [0.99143535]
 [0.99999917]]
1/1 [==============================] - 0s 313us/step - loss: 0.0199 - acc: 1.0000
model.evaluate: [0.019863281399011612, 1.0]
'''