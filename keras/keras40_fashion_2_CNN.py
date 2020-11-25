#fashion_mnist CNN

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input, Conv2D
from tensorflow.keras.layers import Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard 
import matplotlib.pyplot as plt
import numpy as np

# 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#(x_train.shape, x_test.shape) 
#(60000, 28, 28) (10000, 28, 28)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#(y_train.shape, y_test.shape)
#(60000,) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

x_predict = x_train[:10]
y_real = y_train[:10]

# 모델

model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(28,28,1)))
model.add(Conv2D(20, (2,2)))
model.add(Conv2D(30, (2,2)))
model.add(Conv2D(40, (2,2), strides=2))
model.add(MaxPooling2D(pool_size=(2,2))) 
model.add(Flatten()) #3차원을 reshape를 Faltten()으로!
model.add(Dense(100, activation='relu')) 
model.add(Dense(10, activation='softmax'))

model.summary()

# 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', 
                metrics=['acc'])

es = EarlyStopping(monitor='loss', patience=5, mode='auto')
to_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)

model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=1, 
            validation_split=0.5, callbacks=[es, to_hist])

#평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ",loss)
print("acc : ",acc)

y_predict = model.predict(x_predict)

y_predict_recovery = np.argmax(y_predict, axis=1)
y_real_recovery = np.argmax(y_real, axis=1)

print('실제값 : ',y_real_recovery) 
print('예측값 : ',y_predict_recovery) 

