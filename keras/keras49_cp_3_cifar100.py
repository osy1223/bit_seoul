# cifar100 cp CNN

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard 

import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, x_test.shape) 
#(50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape) 
#(50000, 1) (10000, 1)

y_real = y_train[:10]
print(y_real)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

x_predict = x_train[:10] #(10, 32, 32, 3)

# 모델
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(32, 32, 3)))
model.add(Conv2D(20, (2,2)))
model.add(Conv2D(30, (2,2)))
model.add(Conv2D(50, (2,2)))
model.add(Conv2D(30, (2,2)))
model.add(Conv2D(40, (2,2), strides=2))
model.add(MaxPooling2D(pool_size=(2,2))) 
model.add(Flatten()) #3차원을 reshape를 Faltten()으로!
model.add(Dense(50, activation='relu')) 
model.add(Dense(100, activation='softmax'))

model.summary()

# 3. 컴파일, 훈련
modelpath = './model/cifar100-{epoch:02d}-{val_loss:.4f}.hdf5' #현재 모델 경로(study에 model폴더)
#파일명 : epoch:02니깐 2자리 정수 - val_loss .4니깐 소수 4째자리 표기
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', 
        save_best_only=True, mode='auto')

model.compile(loss='categorical_crossentropy', optimizer='adam', 
                metrics=['acc'])

hist = model.fit(x_train, y_train, epochs=30, batch_size=32, 
            verbose=1, validation_split=0.5, callbacks=[es,cp])

# loss = hist.history['loss']
# val_loss = hist.history['val_loss']
# acc = hist.history['acc']
# val_acc = hist.history['val_acc']

#모델+가중치
model.save('./save/cifar100_cnn_model_weights.h5')
model.save_weights('./save/cifar100_cnn_weights.h5')

# 4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", result[0])
print("acc : ", result[1])

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
plt.plot(hist.history['acc'], marker='.', c='red')
plt.plot(hist.history['val_acc'], marker='.', c='blue')
plt.grid()

plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()

'''
loss :  3.9197192192077637
acc :  0.23680000007152557
'''