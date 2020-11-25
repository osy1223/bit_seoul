import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.callbacks import ModelCheckpoint 

# 1. samsung numpy 파일 불러오기 (allow_pickle=True)
# 1. bit numpy 파일 불러오기 (allow_pickle=True)
samsung = np.load('./data/samsung.npy', allow_pickle=True).astype('float32')
bit = np.load('./data/bit.npy', allow_pickle=True).astype('float32')

print(samsung.shape) #(620, 5)
print(bit.shape) #(620, 4)


# 2. x축, y축 자르기

def split_x(seq, size):
    aaa = [] # 임시 리스트
    # i는 0부터 seq사이즈-size까지 반복 
    # (그래야 size만큼씩 온전히 자를 수 있다)
    for i in range(len(seq) - size+1): #seq :전체행의길이, -size+1:자르는길이
        subset = seq[i:(i+size), :] # subset은 i부터 size만큼 배열 저장
        aaa.append(subset) # 배열에 subset을 붙인다
    # print(type(aaa)) # aaa의 타입은 리스트
    return np.array(aaa) #

# 삼성
samsung_x = samsung[:,:4]
samsung_y = samsung[:,4:]

samsung_x=split_x(samsung_x, 5)
samsung_y=samsung_y[5:]

# 비트
bit_x= split_x(bit[:,:3], 5)
bit_y=bit[:, 3:]
bit_y=bit_y[5:]


# 삼성주가 20일걸 예측 하고 싶으면 : 14~19일꺼까지 슬라이싱
samsung_predict = samsung_x[-1]
samsung_x = samsung_x[:-1,:,:]
# samsung_x = samsung_x[:-2,:,:]


bit_predict = bit_x[-1]
bit_x = bit_x[:-1,:,:]

print(samsung_predict.shape) #(5, 4)
print('samsung_x.shape:',samsung_x.shape) #(615, 5, 4)
print('samsung_y.shape:',samsung_y.shape) #(615, 1)
print('bit_x.shape:', bit_x.shape) #(616, 5, 3)
print('bit_y.shape:', bit_y.shape) #(615, 1)

# 3. 데이터 전처리 
# train_test_split
#삼성
samsung_x_train, samsung_x_test, samsung_y_train, samsung_y_test = train_test_split(
    samsung_x, samsung_y, shuffle = True, train_size=0.7)

#비트
bit_x_train, bit_x_test = train_test_split(
    bit_x, shuffle = True, train_size=0.7)

# 4.  모델 만들기
#삼성
input1 = Input(shape=(5,4))
dense1 = LSTM(100, activation='relu')(input1)
dense1 = Dense(600, activation='relu', name='ss1')(dense1)
dense1 = Dense(400, activation='relu', name='ss2')(dense1)
dense1 = Dense(300, activation='relu', name='ss3')(dense1)
output1 = Dense(1, name='ss')(dense1)

#비트
input2 = Input(shape=(5,3))
dense1 = LSTM(100, activation='relu')(input2)
dense1 = Dense(500, activation='relu', name='bit1')(dense1)
dense1 = Dense(600, activation='relu', name='bit2')(dense1)
dense1 = Dense(500, activation='relu', name='bit3')(dense1)
output2 = Dense(1, name='bit')(dense1)

merge = concatenate([output1, output2])

output = Dense(200)(merge)
output3 = Dense(100)(output)
output3 = Dense(1)(output3)

model = Model(inputs=[input1, input2], outputs=output3)

model.summary()

# 5. 컴파일, 훈련
model.compile(loss='mse', 
            optimizer='adam', 
            metrics=['mse'])

es = EarlyStopping(
    monitor='val_loss',
    patience=100,
    mode='auto',
    verbose=2)

modelpath = './model/stock_{epoch:02d}_{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(
    filepath=modelpath,
    monitor='val_loss',
    save_best_only=True,
    mode='auto')

model.fit([samsung_x_train, bit_x_train], samsung_y_train, 
        epochs=10000, 
        batch_size=32, 
        validation_split=0.5,
        verbose=1,
        callbacks=[es,cp])

model.save('./save/stock_model.h5')
model.save_weights('./save/stock_model_weight.h5')


#. 6. 평가
loss, mse = model.evaluate([samsung_x_test, bit_x_test], samsung_y_test)
print('loss:', loss)
print('mse:', mse)

samsung_predict = samsung_predict.reshape(1,samsung_predict.shape[0],4)
bit_predict = bit_predict.reshape(1,5,3)

samsung_pred = model.predict([samsung_predict, bit_predict])
print('samsung_stock_predict:', samsung_pred)

'''
loss: 30759610.0
mse: 30759610.0
samsung_stock_predict: [[68512.805]]
'''