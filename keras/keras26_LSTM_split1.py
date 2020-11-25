import numpy as np

dataset = np.array(range(1,11))
size = 5

# 모델을 구성하시오.
# fit 까지만 할 것
# 7,8,9,10 에 대한 predic 까지

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size +1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset]) #subset
    print(type(aaa))
    return np.array(aaa)

datasets = split_x(dataset, size)
print(datasets)
'''
[[ 1  2  3  4  5]
 [ 2  3  4  5  6]
 [ 3  4  5  6  7]
 [ 4  5  6  7  8]
 [ 5  6  7  8  9]
 [ 6  7  8  9 10]]
'''

#슬라이싱
x = datasets[:, :4] # :모든행, 0부터 4-1까지 열
y = datasets[:, 4] # :모든행, 열이 1개

print(x.shape) # (6, 4)
print(y.shape) # (6,)

# reshape
x = x.reshape(6, 4, 1)

#train_test_split 2개만 쓰기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, train_size=0.7
)

# LSTM 함수형 모델 구성
from tensorflow.keras.models import Model , Sequential
from tensorflow.keras.layers import Dense, LSTM, Input

input1 = Input(shape=(4, 1)) 
dense1 = LSTM(100, activation='relu')(input1)
dense2 = Dense(200, activation='relu')(dense1)
dense3 = Dense(300, activation='relu')(dense2)
output = Dense(1)(dense3)

model = Model(inputs=input1, outputs=output)

model.summary()

# 컴파일, 훈련

#early stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=85, mode='auto')

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x,y, epochs=100, batch_size=1, callbacks=[early_stopping])


# 평가, 예측
x_pred = np.array([7,8,9,10])
x_pred = x_pred.reshape(1, 4, 1)
y_predict = model.predict(x_pred)
print("y_predict :", y_predict)

loss, mse = model.evaluate(x_test, y_test)
print("loss :", loss)
print("mse :", mse)


'''
y_predict : [[10.835928]]
loss :  0.0009002620936371386
mse :  0.0009002620936371386
'''
