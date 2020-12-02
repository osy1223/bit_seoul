import numpy as np

#1. 데이터 준비
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,3,4,5,6,7,8,9,10])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델 구성
model = Sequential()
model.add(Dense(300,input_dim=1, activation='sigmoid'))
model.add(Dense(5000, activation='sigmoid'))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련 (컴퓨터가 알아들을 수 있도록)
model.compile(
    loss='mse', 
    optimizer='adam', 
    metrics=['acc'])

model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
loss, acc = model.evaluate(x,y, batch_size=1)

print("loss : ", loss)
print("acc : ", acc)

y_pred=model.predict(x)
print("결과물 : \n ",y_pred)

'''
activation 비교 

loss :  12.259181022644043
acc :  0.10000000149011612
결과물 :
  [[3.497706]
 [3.497706]
 [3.497706]
 [3.497706]
 [3.497706]
 [3.497706]
 [3.497706]
 [3.497706]
 [3.497706]
 [3.497706]]

loss :  8.250432014465332
acc :  0.10000000149011612
결과물 :
  [[5.479212]
 [5.479212]
 [5.479212]
 [5.479212]
 [5.479212]
 [5.479212]
 [5.479212]
 [5.479212]
 [5.479212]
 [5.479212]]

loss :  8.25
acc :  0.10000000149011612
결과물 :
  [[5.500432]
 [5.500432]
 [5.500432]
 [5.500432]
 [5.500432]
 [5.500432]
 [5.500432]
 [5.500432]
 [5.500432]
 [5.500432]]
'''