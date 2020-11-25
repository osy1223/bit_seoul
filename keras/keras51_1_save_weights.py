# model weights save 

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape) # (60000, 28, 28), (10000, 28, 28)
print(y_train.shape, y_test.shape) # (60000,), (10000,)

# 데이터 전처리 1.OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#전처리 /255
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255. 
# x_test.shape[0] 동일
# .astype : 형변환

# 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Conv2D(10, (2,2), padding='same', input_shape=(28,28,1)))
model.add(Conv2D(20, (2,2), padding='valid')) # (27, 27, 20)
model.add(Conv2D(30, (3,3)))
model.add(Conv2D(40, (2,2), strides=2)) #strides : 옮겨가는 값 (2칸씩 옮길꺼야)
model.add(MaxPooling2D(pool_size=2)) #pool_size=2 디폴트 값
model.add(Flatten())
model.add(Dense(100, activation='relu')) #Dense activation 디폴트 값 : relu
model.add(Dense(10, activation='softmax')) # 총 결과값 : 10(output)

model.summary()

model.save("./save/model_test02_1.h5")

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
cp = ModelCheckpoint(filepath='./model/minist-{epoch:02d}-{val_loss:.4f}.hdf5', monitor='val_loss', 
        save_best_only=True, mode='auto')


model.compile(loss='categorical_crossentropy', optimizer='adam', 
                metrics=['acc'])

hist = model.fit(x_train, y_train, epochs=30, batch_size=32, 
            verbose=1, validation_split=0.5, callbacks=[es])

# 모델 + 가중치 저장
model.save("./save/model_test02_2.h5")

# 가중치 저장
model.save_weights('./save/weight_test02_3.h5')

# 4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", result[0])
print("acc : ", result[1])

'''
loss :  0.09647425264120102
acc :  0.9837999939918518
'''

