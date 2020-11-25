# 모델 load 해서 완성!!

import numpy as np

dataset = np.array(range(1,101))
size = 5

# Dense 모델을 구성하시오.
# fit 까지만 할 것
# predic 까지

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size +1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset]) #subset
    print(type(aaa))
    return np.array(aaa)

datasets = split_x(dataset, size)

x = datasets[:, 0:4]
y = datasets[:, 4]

print(x.shape) # (96, 4)
print(y.shape) # (96,)

x = x.reshape(x.shape[0], x.shape[1], 1) #3차원
print(x.shape) # (96, 4, 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7
)

# 모델 불러오기 + 커스터마이징
from tensorflow.keras.models import Model , load_model
from tensorflow.keras.layers import Dense, Input

model = load_model('./save/keras28.h5')
model.add(Dense(10, name='king1'))
model.add(Dense(1, name='king2'))

#model.summary()

# 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=85, mode='auto')

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x_train, y_train, epochs=100, batch_size=1, callbacks=[early_stopping],
    validation_split=0.3,)

# 예측, 평가
x_predict = np.array([97,98,99,100]) #(4,)
x_predict = x_predict.reshape(1, 4, 1)

y_predict = model.predict(x_predict)
print("y_predict :", y_predict) 

loss, mse = model.evaluate(x_test, y_test)
print("loss, mse :", loss, mse)

'''
y_predict : [[100.990585]] 원하는 값 101
loss, mse : 0.06965656578540802 0.06965656578540802
'''
