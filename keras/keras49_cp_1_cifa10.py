#cifar10 cp CNN

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical #이미지 분류 작업
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard 
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print(x_train.shape, x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3)
# print(y_train.shape, y_test.shape) #(50000, 1) (10000, 1)

# 데이터 전처리 1.OneHotEncoding . 라벨링
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(y_train.shape, y_test.shape) #(50000, 10) (10000, 10)

#전처리 /255
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255. 

x_predict = x_train[:10]
y_real = y_train[:10]

# x_predict = x_predict.reshape(10, 32, 32, 3).astype('float32')/255.

# 모델 구성
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(32,32,3)))
model.add(Conv2D(20, (2,2)))
model.add(Conv2D(30, (2,2)))
model.add(Conv2D(40, (2,2), strides=2))
model.add(MaxPooling2D(pool_size=(2,2))) 
model.add(Flatten()) #3차원을 reshape를 Faltten()으로!
model.add(Dense(100, activation='relu')) 
model.add(Dense(10, activation='softmax'))

model.summary()

# 3. 컴파일, 훈련
modelpath = './model/cifar10-{epoch:02d}-{val_loss:.4f}.hdf5' #현재 모델 경로(study에 model폴더)
#파일명 : epoch:02니깐 2자리 정수 - val_loss .4니깐 소수 4째자리 표기
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=5, mode='auto')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', 
        save_best_only=True, mode='auto')

model.compile(loss='categorical_crossentropy', optimizer='adam', 
                metrics=['acc'])

hist = model.fit(x_train, y_train, epochs=30, batch_size=32, 
            verbose=1, validation_split=0.5, callbacks=[es,cp])

#모델+가중치
model.save('./save/cifar10_cnn_model_weights.h5')
model.save_weights('./save/cifar10_cnn_weights.h5')


# 4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", result[0])
print("acc : ", result[1])

y_predict = model.predict(x_predict)

y_predict = np.argmax(y_predict, axis=1)
y_real = np.argmax(y_real, axis=1)

print('실제값 : ',y_real) 
print('예측값 : ',y_predict) 

# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6)) #단위 무엇인지 찾아볼것!


plt.subplot(2,1,1) #(2행 1열에서 1번째 그림)

plt.plot(hist.history['loss'], marker='.', c='red', label='loss') 
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')

plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right') 


plt.subplot(2,1,2) #(2행 1열에서 2번째 그림)
plt.plot(hist.history['acc'], marker='.', c='red')
plt.plot(hist.history['val_acc'], marker='.', c='blue')
plt.grid()

plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()

'''
loss :  1.692379117012024
acc :  0.6126999855041504
실제값 :  [6 9 9 4 1 1 2 7 8 3]
예측값 :  [6 9 9 4 1 1 2 7 8 3]
'''