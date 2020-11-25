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
gold = np.load('./data/gold.npy', allow_pickle=True).astype('float32')
kosdaq = np.load('./data/kosdaq.npy', allow_pickle=True).astype('float32')

print(samsung.shape) #(620, 5)
print(bit.shape) #(620, 4)
print(gold.shape) #(620, 6)
print(kosdaq.shape) #(620, 3)


# 2. x축, y축 자르기
def split_x(seq, size):
    aaa = [] # 임시 리스트
    # i는 0부터 seq사이즈-size까지 반복 
    # (그래야 size만큼씩 온전히 자를 수 있다)
    for i in range(len(seq) - size+1): #seq :전체행의길이, -size+1:자르는길이
        subset = seq[i:(i+size), :] # subset은 i부터 size만큼 배열 저장
        aaa.append(subset) # 배열에 subset을 붙인다
    # print(type(aaa)) # aaa의 타입은 리스트
    return np.array(aaa) 

# 삼성 (620, 5)
samsung_x = samsung[:,:4]
samsung_y = samsung[:,4:]

samsung_x=split_x(samsung_x, 5)
samsung_y=samsung_y[5:]

# 비트 (620, 4)
bit_x= split_x(bit[:,:3], 5)
bit_y=bit[:, 3:]
bit_y=bit_y[5:]

# 골드 (620, 6)
gold_x = split_x(gold[:, :5], 5)
gold_y = gold[:, 5:]
gold_y = gold_y[5:]

# 코스닥 (620, 3)
kosdaq_x = split_x(kosdaq[:, :2], 5)
kosdaq_y = kosdaq[:, 3:]
kosdaq_y = kosdaq_y[5:]

# 삼성 23일 시가 예측 하고 싶어서
samsung_predict = samsung_x[-1]
# samsung_x = samsung_x[:-1,:,:]
samsung_x = samsung_x[:-2,:,:]
samsung_y = samsung_y[:-1,:]



print(samsung_x.shape)
print(samsung_y.shape)

print(bit_x.shape)
print(bit_y.shape)

print(gold_x.shape)
print(gold_y.shape)

print(kosdaq_x.shape)
print(kosdaq_y.shape)



# 예측값
bit_predict=bit_x[-1]
print(bit_predict)
bit_x=bit_x[:-1,:,:]
print(bit_predict)

gold_predict=gold_x[-1]
kosdaq_predict=kosdaq_x[-1]

print(samsung_predict.shape) #(5, 4)
print('samsung_x.shape:',samsung_x.shape) #(615, 5, 4)

print('bit_x.shape:', bit_x.shape) #(616, 5, 3)

print('gdol_x.shape:',gold_x.shape) #(616, 5, 5)

print('kosdaq_x.sahpe:',kosdaq_x.shape) #(616, 5, 2)


# 3. 데이터 전처리 
# train_test_split
#삼성
samsung_x_train, samsung_x_test, samsung_y_train, samsung_y_test = train_test_split(
    samsung_x, samsung_y, shuffle = True, train_size=0.7)

#비트
bit_x_train, bit_x_test = train_test_split(
    bit_x, shuffle = True, train_size=0.7)

#골드
gold_x_train, gold_x_test = train_test_split(
    gold_x, shuffle = True, train_size=0.7)

#코스닥
kosdaq_x_train, kosdaq_x_test = train_test_split(
    kosdaq_x, shuffle = True, train_size=0.7)


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


#골드
input3 = Input(shape=(5,5))
dense1 = LSTM(100, activation='relu')(input3)
dense1 = Dense(500, activation='relu', name='gold1')(dense1)
dense1 = Dense(600, activation='relu', name='gold2')(dense1)
dense1 = Dense(500, activation='relu', name='gold3')(dense1)
output3 = Dense(1, name='gold')(dense1)

#코스닥
input4 = Input(shape=(5,2))
dense1 = LSTM(100, activation='relu')(input4)
dense1 = Dense(500, activation='relu', name='kosdaq1')(dense1)
dense1 = Dense(600, activation='relu', name='kosdaq2')(dense1)
dense1 = Dense(500, activation='relu', name='kosdaq3')(dense1)
output4 = Dense(1, name='kosdaq')(dense1)

merge = concatenate([output1, output2, output3, output4])

output = Dense(200)(merge)
output = Dense(100)(output)
output = Dense(1)(output)

model = Model(inputs=[input1, input2, input3, input4], outputs=output)

# model.summary()


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

model.fit([samsung_x_train, bit_x_train, gold_x_train, kosdaq_x_train], samsung_y_train, 
        epochs=10000, #튜닝
        batch_size=32, 
        validation_split=0.5,
        verbose=1,
        callbacks=[es,cp])

model.save('./save/stock_model.h5')
model.save_weights('./save/stock_model_weight.h5')


#. 6. 평가
loss, mse = model.evaluate([samsung_x_test, bit_x_test, gold_x_test, kosdaq_x_test], samsung_y_test)
print('loss:', loss)
print('mse:', mse)

samsung_predict = samsung_predict.reshape(1,samsung_predict.shape[0],4)
bit_predict = bit_predict.reshape(1,5,3)
gold_predict = gold_predict.reshape(1,5,5)
kosdaq_predict = kosdaq_predict.reshape(1,5,2)


samsung_pred = model.predict([samsung_predict, bit_predict, gold_predict, kosdaq_predict])
print('samsung_stock_predict:', samsung_pred)

'''
loss: 36733308.0
mse: 36733308.0
samsung_stock_predict: [[52690.836]]
'''