# cifar 10 data load

# from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard 
import matplotlib.pyplot as plt
import numpy as np

################### 데이터 ########################

# cifar 10 data load
c10_x_train = np.load('./data/c10_x_train.npy')
c10_x_test = np.load('./data/c10_x_test.npy')
c10_y_train = np.load('./data/c10_y_train.npy')
c10_y_test = np.load('./data/c10_y_test.npy')

# OneHotEncoding . 라벨링
c10_y_train = to_categorical(c10_y_train)
c10_y_test = to_categorical(c10_y_test)

#전처리 CNN 4차원이라 괜찮습니다~
c10_x_train = c10_x_train.astype('float32')/255.
c10_x_test = c10_x_test.astype('float32')/255. 

x_predict = c10_x_train[:10]
y_real = c10_y_train[:10]


################### 1. load_model ########################

#3. 컴파일, 훈련
from tensorflow.keras.models import load_model
model1 = load_model('./save/model_test02_2.h5')

#4. 평가, 예측
result1 = model1.evaluate(c10_x_test, c10_y_test, batch_size=32)
print("r1 loss : ", result1[0])
print("r1 acc : ", result1[1])


############## 2. load_model ModelCheckPoint #############
from tensorflow.keras.models import load_model
model2 = load_model('./model/cifar10-30-0.1863.hdf5')

#4. 평가, 예측
result2 = model2.evaluate(c10_x_test, c10_y_test, batch_size=32)
print("r2 loss : ", result2[0])
print("r2 accuracy : ", result2[1])

############### 3. load_weights ##################
# 2. 모델
model3 = Sequential()
model3.add(Conv2D(10, (2,2), padding='same', input_shape=(32,32,3)))
model3.add(Conv2D(20, (2,2), padding='valid'))
model3.add(Conv2D(30, (3,3)))
model3.add(Conv2D(40, (2,2), strides=2))
model3.add(MaxPooling2D(pool_size=2))
model3.add(Flatten())
model3.add(Dense(100, activation='relu'))
model3.add(Dense(10, activation='softmax'))


# 3. 컴파일
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model3.load_weights('./save/weight_test02_3.h5')

#4. 평가, 예측
result3 = model3.evaluate(c10_x_test, c10_y_test, batch_size=32)
print("r3 loss : ", result3[0])
print("r3 acc : ", result3[1])