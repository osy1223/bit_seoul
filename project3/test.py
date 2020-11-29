import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import matplotlib.pyplot as plt

############################################ 데이터 불러오기
x = np.load('./project3/merge_x.npy')
y = np.load('./project3/merge_y.npy')

# print(x)
# print(y)

############################################ train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=44, shuffle=True, train_size=0.8
)

print(x_train.shape) #(7608, 15)
print(x_test.shape) #(1902, 15)

# print(y_train)
# print(y_test)

############################################ reshape
x_train = x_train.reshape(7608, 5, 3)
x_test = x_test.reshape(1902, 5, 3)


############################################ 모델
model = Sequential()
model.add(LSTM(500, input_shape=(5,3)))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(1))

model.summary()

############################################ 컴파일
es = EarlyStopping(monitor='loss', patience=100, mode='auto')
hist = TensorBoard(log_dir='graph', histogram_freq=0,
        write_graph=True, write_images=True)

model.compile(loss='mse', optimizer='adam',
            metrics=['mse'])
model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=1,
            validation_split=0.5, callbacks=[es, hist])

############################################ 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ",loss)
print("mse : ",mse)

'''
loss :  99.37525939941406
mse :  99.37525939941406

loss :  27.20890998840332
mse :  27.20890998840332
'''

############################################ 시각화
# plt.figure(figsize=(10,6))

# plt.subplot(2,1,1) #(2행 1열에서 1번째 그림)
# plt.plot(hist.history['loss'], marker='.', c='red', label='loss') # x값과, y값이 들어가야 합니다
# #x는 안넣어도 순서대로 list 형식으로 저장되서 안 넣어줬습니다
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
# plt.grid()
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc='upper right') #우측 상단에 legend(label 2개 loss랑 val_loss) 표시

# plt.subplot(2,1,2) #(2행 1열에서 2번째 그림)
# plt.plot(hist.history['mae'], marker='.', c='red')
# plt.plot(hist.history['val_mae'], marker='.', c='blue')
# plt.grid()
# plt.title('mae')
# plt.ylabel('mae')
# plt.xlabel('epoch')
# plt.legend(['mae', 'val_mae'])

plt.show()