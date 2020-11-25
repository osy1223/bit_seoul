#boston DNN cp

import numpy as np
from sklearn.datasets import load_boston
dataset = load_boston()
x = dataset.data
y = dataset.target

print(x.shape) #(506, 13)
print(y.shape) #(506,)

# 데이터 전처리
# train, test 분리. validation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7
)

# Scaler
scaler = StandardScaler()
scaler.fit(x_train) # fit하고
x_train = scaler.transform(x_train) # 사용할 수 있게 바꿔서 저장
x_test = scaler.transform(x_test)

# reshape?
# x가 2차원이라, 별도의 reshape 없이 그대로 x를 사용한다

# DNN 모델, output 레이어는 1개
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],) ))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1) )
model.summary()

# 3. 컴파일, 훈련
modelpath = './model/boston-{epoch:02d}-{val_loss:.4f}.hdf5' #현재 모델 경로(study에 model폴더)
#파일명 : epoch:02니깐 2자리 정수 - val_loss .4니깐 소수 4째자리 표기
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='loss', patience=5, mode='auto')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', 
        save_best_only=True, mode='auto')

model.compile(loss='mse', optimizer='adam', 
                metrics=['mae'])

hist = model.fit(x_train, y_train, epochs=100, batch_size=32, 
            verbose=1, validation_split=0.5, callbacks=[es,cp])

# 4. 평가, 예측
loss,mae = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("mae : " ,mae)

y_predict = model.predict(x_test) # 평가 데이터 다시 넣어 예측값 만들기

# 사용자정의 RMSE 함수
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE:", RMSE(y_test, y_predict))


# 사용자정의 R2 함수
# 사이킷런의 metrics에서 r2_score를 불러온다
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2:", r2)

# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6)) #단위 무엇인지 찾아볼것!

plt.subplot(2,1,1) #(2행 1열에서 1번째 그림)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss') # x값과, y값이 들어가야 합니다
#x는 안넣어도 순서대로 list 형식으로 저장되서 안 넣어줬습니다
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right') #우측 상단에 legend(label 2개 loss랑 val_loss) 표시

plt.subplot(2,1,2) #(2행 1열에서 2번째 그림)
plt.plot(hist.history['mae'], marker='.', c='red')
plt.plot(hist.history['val_mae'], marker='.', c='blue')
plt.grid()

plt.title('mae')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['mae', 'val_mae'])

plt.show()

'''
loss :  12.266814231872559
mae :  2.689046859741211
RMSE: 3.502401230658142
R2: 0.8646993729543531
'''