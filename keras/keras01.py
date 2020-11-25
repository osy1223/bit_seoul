import numpy as np

#1. 데이터 준비
x=np.array([1,2,3,4,5])
y=np.array([1,2,3,4,5])

from tensorflow.keras.models import Sequential
#텐서플로우 케라스 모델 안에 시퀀스(순차적)를 가져오겠다.
from tensorflow.keras.layers import Dense
#텐서플로우 케라스 레이어(층) 안에 댄스층을 가져오겠다.

#2. 모델 구성
model = Sequential()
#Sequential 순차적(위->아래)로 가면서 연산(y=wx+b)
model.add(Dense(300,input_dim=1))
#Dense 3층을 쌓겠다. dim 1개 입력(디멘션 input 행이 1차원이라 1개)
model.add(Dense(5000))
model.add(Dense(30))
model.add(Dense(7))
model.add(Dense(1))

#3. 컴파일, 훈련 (컴퓨터가 알아들을 수 있도록)
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
#optimizer 파라미터 최적화 (adam 방향성 + 스텝사이즈)
#metrics 평가기준 (일반적으로 ‘accuracy’)

model.fit(x, y, epochs=1000, batch_size=1)
#batch_size=1 1번씩 잘라서, epochs=100 100번 훈련시키겠다.

#4. 평가, 예측
loss, acc = model.evaluate(x,y, batch_size=1)

print("loss : ", loss)
print("acc : ", acc)