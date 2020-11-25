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

print(x.shape) #(96, 4)
print(y.shape) #(96,)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7
)

print(x_train.shape) #(67, 4, 1)


# LSTM 함수형 모델 구성
from tensorflow.keras.models import Model , Sequential
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Dense(30, activation='relu', input_dim=4)) #column 개수=?
model.add(Dense(70, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()

# 컴파일, 훈련

#early stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=85, mode='auto')

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x,y, epochs=100, batch_size=1)


# 평가, 예측
x_pred = np.array([97,98,99,100])
print(x_pred.shape)
x_pred = x_pred.reshape(1, 4)

y_predict = model.predict(x_pred)
print("y_pred :", y_predict)

loss, mse = model.evaluate(x_test, y_test)
print("loss, mse :", loss,mse)

'''
y_pred : [[101.04716]]
loss, mse : 0.0011735953157767653 0.0011735953157767653
'''
