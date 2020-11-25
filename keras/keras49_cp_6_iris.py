# iris cnn cp

from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.layers import Conv2D, Flatten, MaxPool2D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

import matplotlib.pyplot as plt
import numpy as np

# 데이터
iris = load_iris()
x = iris.data
y = iris.target

print(x.shape) #(150, 4)
print(y.shape) #(150,)

# 데이터 전처리
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
x = x.reshape(150, 2, 2, 1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7
)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_predict = x_train[:10]
y_real = y_train[:10]

# 모델링
model = Sequential()
model.add(Conv2D(100, (2,2), padding='same', input_shape=(x_train.shape[1],2,1)))
model.add(Conv2D(300, (1,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(20))
model.add(Dense(3, activation='softmax'))

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
loss :  0.18260696530342102
acc :  0.9111111164093018
'''