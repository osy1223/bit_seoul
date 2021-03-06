#1. 데이터
import numpy as np

#input 2개
x1 = np.array((range(1,101), range(711, 811), range(100)))
x2 = np.array((range(4,104), range(761, 861), range(100)))

# output 3개 
y1 = np.array((range(101,201), range(311,411), range(100)))
y2 = np.array((range(501,601), range(431,531), range(100,200)))
y3 = np.array((range(501,601), range(431,531), range(100,200)))

x1=np.transpose(x1)
x2=np.transpose(x2)

y1=np.transpose(y1)
y2=np.transpose(y2)
y3=np.transpose(y3)

#train_test_split 2개만 쓰기
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1,y1, shuffle=True, train_size=0.7
)

x2_train, x2_test, y2_train, y2_test, y3_train, y3_test = train_test_split(
    x2,y2,y3, shuffle=True, train_size=0.7
)

# from sklearn.model_selection import train_test_split
# y3_train, y3_test = train_test_split(
#     y3, shuffle=True, train_size=0.7
# )

#2. 함수형 모델 2개 구성

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input

# 모델1
input1 = Input(shape=(3,))
dense1_1 = Dense(10, activation='relu', name='king1')(input1)
dense1_2 = Dense(7, activation='relu', name='king2')(dense1_1)
dense1_3 = Dense(5, activation='relu', name='king3')(dense1_2)
output1 = Dense(3, activation='linear', name='king4')(dense1_3)

# 모델2
input2 = Input(shape=(3,))
dense2_1 = Dense(15, activation='relu', name='queen1')(input2)
dense2_2 = Dense(11, activation='relu', name='queen2')(dense2_1)
output2 = Dense(3, activation='linear', name='queen3')(dense2_2) #activation='linear'인 상태

#----------------------------------------------------------------------------------

#모델 병합, concatenate
from tensorflow.keras.layers import Concatenate, concatenate

merge1 = Concatenate()([output1, output2])
middle1 = Dense(30)(merge1)
middle1 = Dense(7)(middle1)
middle1 = Dense(11)(middle1)

################# output 모델 구성 (분기)
output1 = Dense(30)(middle1)
output1 = Dense(7)(output1)
output1 = Dense(3)(output1)

output2 = Dense(15)(middle1)
output2_1 = Dense(14)(output2)
output2_3 = Dense(11)(output2_1)
output2_4 = Dense(3)(output2_3)

output3 = Dense(30)(middle1)
output3_1 = Dense(20)(output3)
output3_3 = Dense(10)(output3_1)
output3_4 = Dense(3)(output3_3)

# 모델 정의
model = Model(inputs = [input1, input2], 
              outputs = [output1, output2_4, output3_4])

model.summary()
#concatenate는 연산하지 않는다 

# 컴파일 , 훈련
model.compile(
    loss='mse', 
    optimizer='adam', 
    metrics=['mse'])

model.fit([x1_train, x2_train], [y1_train, y2_train, y3_train], 
    epochs=256, 
    batch_size=32,
    validation_split=0.5,
    verbose=1)

# 평가
result = model.evaluate([x1_test, x2_test], [y1_test, y2_test, y3_test], batch_size=7)
print("result :", result)


#model에 맞춰서 input을 주어야 한다
y_pred_1, y_pred_2, y_pred_3 = model.predict([x1_test, x2_test])


# RMSE
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
    # predict해서 나온 값과 원래 y_test 값을 비교해서 RMSE로 나오게 하겠다

print("\nRMSE_1: ", RMSE(y1_test, y_pred_1))
print("RMSE_2: ", RMSE(y2_test, y_pred_2))
print("RMSE_3: ", RMSE(y3_test, y_pred_3), "\n")


# R2는 함수 제공
from sklearn.metrics import r2_score
print("R2_1: ", r2_score(y1_test, y_pred_1))
print("R2_2: ", r2_score(y2_test, y_pred_2))
print("R2_3: ", r2_score(y3_test, y_pred_3), "\n")

'''
output이 3개 -> loss 3개, mse3개가 출력되고, 전체 loss 포함해서 7개 출력
result : [875.205078125,
         732.5170288085938, 
         101.29524993896484, 
         41.39277648925781, 
         732.5170288085938, 
         101.29524993896484,
          41.39277648925781]

RMSE_1:  6.772575008082211
RMSE_2:  17.218078365941324
RMSE_3:  16.932915231607936 

R2_1:  0.951458394313326
R2_2:  0.6129376670388212
R2_3:  0.6256524301403987 
'''