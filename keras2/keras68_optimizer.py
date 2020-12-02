import numpy as np

#1. 데이터 준비
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,3,4,5,6,7,8,9,10])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델 구성
model = Sequential()
model.add(Dense(300, input_dim=1))
model.add(Dense(5000))
model.add(Dense(30))
model.add(Dense(7))
model.add(Dense(1))

#3. 컴파일, 훈련 (컴퓨터가 알아들을 수 있도록)
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

# (0.001이 디폴트)
# optimizer = Adam(lr=0.001)
# optimizer = Adam(lr=0.1) #아담의 러닝메이트를 0.1로 튜닝
# optimizer = Adadelta(lr=0.1)
# optimizer = Adamax(lr=0.1)
# optimizer = Adagrad(lr=0.1)
# optimizer = RMSprop(lr=0.1)
# optimizer = SGD(learning_rate=0.1)
optimizer = Nadam(lr=0.1)

model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
model.fit(x, y, epochs=10, batch_size=1)

#4. 평가, 예측
loss, mse = model.evaluate(x,y, batch_size=1)
y_pred=model.predict([11])
print("loss : ", loss,'\n',"결과물 : ", y_pred)

'''
optimizer = Adam(lr=0.001)
loss :  0.020883943885564804
결과물 :  [[10.93927]]

optimizer = Adam(lr=0.1)
loss :  291845.0
결과물 :  [[-1031.0316]]

optimizer = Adadelta(lr=0.1)
loss :  0.08006033301353455
결과물 :  [[10.489023]]

optimizer = Adamax(lr=0.1)
loss :  46965.92578125
결과물 :  [[-277.47177]]

optimizer = Adagrad(lr=0.1)
loss :  3119.68408203125
결과물 :  [[79.5444]]

optimizer = RMSprop(lr=0.1)
loss :  105035544.0
결과물 :  [[-19017.049]]

optimizer = SGD(learning_rate=0.1)
loss :  nan
결과물 :  [[nan]]

optimizer = Nadam(lr=0.1)
loss :  424435.6875
결과물 :  [[683.6954]]
'''