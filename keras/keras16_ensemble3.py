#1. 데이터
import numpy as np

#input 2개
x1 = np.array((range(1,101), range(711, 811), range(100)))
x2 = np.array((range(1,101), range(761, 861), range(100)))

# output 1개 
y1 = np.array((range(101,201), range(311,411), range(100)))


x1 = np.transpose(x1)
x2 = np.transpose(x2)

y1 = np.transpose(y1)

#train_test_split 2개만 쓰기
from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test = train_test_split(
    x1,x2, shuffle=True, train_size=0.7
)

from sklearn.model_selection import train_test_split
y1_train, y1_test = train_test_split(
    y1, shuffle=True, train_size=0.7
)

#2. 함수형 모델 2개 구성

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input

# 모델1
input1 = Input(shape=(3,))
dense1_1 = Dense(1000, activation='relu', name='king1')(input1)
dense1_2 = Dense(700, activation='relu', name='king2')(dense1_1)
dense1_3 = Dense(5000, activation='relu', name='king3')(dense1_2)
dense1_4 = Dense(300, activation='relu', name='king4')(dense1_3)
dense1_5 = Dense(500, activation='relu', name='king5')(dense1_4)
output1 = Dense(3, activation='linear', name='king')(dense1_5)

# 모델2
input2 = Input(shape=(3,))
dense2_1 = Dense(100, activation='relu', name='queen1')(input2)
dense2_2 = Dense(2000, activation='relu', name='queen2')(dense2_1)
dense2_3 = Dense(2000, activation='relu', name='queen3')(dense2_2)
dense2_4 = Dense(2000, activation='relu', name='queen4')(dense2_3)
output2 = Dense(300, activation='linear', name='queen')(dense2_4) #activation='linear'인 상태

#----------------------------------------------------------------------------------

#모델 병합, concatenate
from tensorflow.keras.layers import Concatenate, concatenate

merge1 = Concatenate()([output1, output2])

middle1 = Dense(3000)(merge1)
middle2 = Dense(7000)(middle1)
middle3 = Dense(1000)(middle2)
middle4 = Dense(100)(middle3)
middle5 = Dense(2000)(middle4)
middle6 = Dense(3000)(middle5)
middle7 = Dense(1000)(middle6)
middle8 = Dense(5000)(middle7)
middle9 = Dense(1000)(middle8)


################# output 모델 구성 (output1개라 분기 필요없음)
output1 = Dense(30)(middle9)
output1_1 = Dense(7)(output1)
output1_2 = Dense(3)(output1_1)
output1_3 = Dense(3)(output1_2)
output1_4 = Dense(3)(output1_3)
output1_5 = Dense(3)(output1_4)
output1_6 = Dense(3)(output1_5)
output1_7 = Dense(3)(output1_6)
output1_8 = Dense(3)(output1_7)
output1_9 = Dense(3)(output1_8)

# 모델 정의
model = Model(inputs = [input1, input2], 
              outputs = output1_9)

model.summary()

# 컴파일 , 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train], y1_train, epochs=100, batch_size=8,
          validation_split=0.25, verbose=1)

# 평가
result = model.evaluate([x1_test, x2_test], y1_test, batch_size=8)

print("result :", result)

y1_pred = model.predict([x1_test, x2_test])
print("결과물 : ", y1_pred)

#RMSE 함수 사용자 정의
from sklearn.metrics import mean_squared_error
def RMSE(y1_test, y1_pred):
    return np.sqrt(mean_squared_error(y1_test, y1_pred))
print("RMSE :", RMSE(y1_test, y1_pred))

#R2 함수
from sklearn.metrics import r2_score
r2 = r2_score(y1_test, y1_pred)
print("R2 : ",r2)


# 파라미터 튠 까지 하세요~_~