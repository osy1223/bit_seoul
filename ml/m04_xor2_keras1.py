# SVC

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
model.add(Dense(1, input_dim=2, activation='sigmoid'))

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

# acc2 = accuracy_score(y_data, y_predict)
# print('accuracy_score:',acc2)

'''
[[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측결과: [[0.54139507]
 [0.39140284]
 [0.7644574 ]
 [0.638739  ]]
 
 model.evaluate: [0.6086081266403198, 0.5]
'''