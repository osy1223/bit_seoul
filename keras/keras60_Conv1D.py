#Conv1D

from tensorflow.keras.layers import Conv1D, Dense, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling1D
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np

# 데이터
a = np.array(range(1,101)) #(100,)
size =5

# split_x 함수
def split_x(seq, size):
    aaa = []
    for i in range(len(seq)-size+1):
        subset = seq[i:(i+size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(a, size)

x = dataset[:,:4]
y = dataset[:,4:]
print('x.shape :', x.shape) #x.shape : (96, 4) 
print('y.shape : ',y.shape) #y.shape :  (96, 1)

# train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True
)
print("after_x_train_shape : ", x_train.shape) #(67, 4)
print("after x_test.shape", x_test.shape) #(29, 4)

# # 데이터 전처리
# scaler = StandardScaler()
# scaler.fit(x_train)

# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# reshape
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
print("reshape x:", x_train.shape, x_test.shape)
# (67, 4, 1) (29, 4, 1)

# Conv1D 모델 구성
model = Sequential()
model.add(Conv1D(100, 3, input_shape=(x_train.shape[1],1))) #(4,1)
model.add(Conv1D(50, 2, padding='same'))
model.add(Flatten())
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

# 컴파일, 훈련
model.compile(
    loss='mse', 
    optimizer='adam',
    metrics=['mse'])

es = EarlyStopping(
    monitor='val_loss', 
    patience=7, 
    mode='auto')
    
cp = ModelCheckpoint(
    filepath='./model/keras60_{epoch:02d}_{val_loss:.4f}.hdf5',
    monitor='val_loss',
    save_best_only=True,
    mode='auto')

hist = TensorBoard(
    log_dir='graph', 
    histogram_freq=0,
    write_graph=True, write_images=True)

model.fit(x_train, y_train, 
    epochs=1000, 
    batch_size=32, 
    verbose=1,
    validation_split=0.5, 
    callbacks=[es,cp,hist])

# 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("mse : ", mse)

y_predict = model.predict(x_test)
# y_predict = np.argmax(y_predict, axis=1)
print("y_test.shape:", y_test.shape) #(29, 1)
print("y_predict.shape:", y_predict.shape) #(29, 1)

y_recovery = y_test

# 사용자정의 RMSE 함수
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE:", RMSE(y_recovery, y_predict))


# 사용자정의 R2 함수
# 사이킷런의 metrics에서 r2_score를 불러온다
from sklearn.metrics import r2_score
r2 = r2_score(y_recovery, y_predict)
print("R2:", r2)

x_predict = np.array([97,98,99,100])
x_predict = x_predict.reshape(1,4 ,1) 

y_predict = model.predict(x_predict)
print("y_predict:\n", y_predict)
# predict = 101하나만 나오도록

'''
loss :  0.9577646851539612
mse :  0.9577646851539612
y_test.shape: (29, 1)
y_predict.shape: (29, 1)
RMSE: 0.9786545035281118
R2: 0.9988908991444373
y_predict:
 [[100.89468]]
'''