# model1, model2, model3

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical #이미지 분류 작업
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard 
import matplotlib.pyplot as plt
import numpy as np

################### 데이터 ########################

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)
#(50000, 32, 32, 3) (50000, 1)

# OneHotEncoding . 라벨링
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#전처리 CNN 4차원이라 괜찮습니다~
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255. 

x_predict = x_train[:10]
y_real = y_train[:10]


################### 1. load_model ########################

#3. 컴파일, 훈련
from tensorflow.keras.models import load_model
model1 = load_model('./save/cifar10_cnn_model_weights.h5')

#4. 평가, 예측
result1 = model1.evaluate(x_test, y_test, batch_size=32)
print("r1 loss : ", result1[0])
print("r1 acc : ", result1[1])


############## 2. load_model ModelCheckPoint #############
from tensorflow.keras.models import load_model
model2 = load_model('./model/cifar10-03-1.0974.hdf5')

#4. 평가, 예측
result2 = model2.evaluate(x_test, y_test, batch_size=32)
print("r2 loss : ", result2[0])
print("r2 accuracy : ", result2[1])

############### 3. load_weights ##################
# 2. 모델
model3 = Sequential()
model3.add(Conv2D(10, (2,2), input_shape=(32,32,3)))
model3.add(Conv2D(20, (2,2)))
model3.add(Conv2D(30, (2,2)))
model3.add(Conv2D(40, (2,2), strides=2))
model3.add(MaxPooling2D(pool_size=(2,2))) 
model3.add(Flatten()) #3차원을 reshape를 Faltten()으로!
model3.add(Dense(100, activation='relu')) 
model3.add(Dense(10, activation='softmax'))


# 3. 컴파일
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model3.load_weights('./save/cifar10_cnn_weights.h5')

#4. 평가, 예측
result3 = model3.evaluate(x_test, y_test, batch_size=32)
print("r3 loss : ", result3[0])
print("r3 acc : ", result3[1])

'''
r1 loss :  1.7595691680908203
r1 acc :  0.6116999983787537

r2 loss :  1.0874758958816528
r2 accuracy :  0.6299999952316284  

r3 loss :  1.7595691680908203
r3 acc :  0.6116999983787537
'''