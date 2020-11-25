# fashion data load

# from tensorflow.keras.datasets import fashion
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard 
import matplotlib.pyplot as plt
import numpy as np

################### 데이터 ########################

# fm data load
fm_x_train = np.load('./data/fm_x_train.npy')
fm_x_test = np.load('./data/fm_x_test.npy')
fm_y_train = np.load('./data/fm_y_train.npy')
fm_y_test = np.load('./data/fm_y_test.npy')

# OneHotEncoding . 라벨링
fm_y_train = to_categorical(fm_y_train)
fm_y_test = to_categorical(fm_y_test)

#전처리 CNN 4차원이라 괜찮습니다~
fm_x_train = fm_x_train.astype('float32')/255.
fm_x_test = fm_x_test.astype('float32')/255. 

x_predict = fm_x_train[:10]
y_real = fm_y_train[:10]


################### 1. load_model ########################

#3. 컴파일, 훈련
from tensorflow.keras.models import load_model
model1 = load_model('./save/model_test02_2.h5')

#4. 평가, 예측
result1 = model1.evaluate(fm_x_test, fm_y_test, batch_size=32)
print("r1 loss : ", result1[0])
print("r1 acc : ", result1[1])


############## 2. load_model ModelCheckPoint #############
from tensorflow.keras.models import load_model
model2 = load_model('./model/fashion-04-0.3096.hdf5')

#4. 평가, 예측
result2 = model2.evaluate(fm_x_test, fm_y_test, batch_size=32)
print("r2 loss : ", result2[0])
print("r2 accuracy : ", result2[1])

############### 3. load_weights ##################
# 2. 모델
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(28,28,1)))
model.add(Conv2D(20, (2,2)))
model.add(Conv2D(30, (2,2)))
model.add(Conv2D(40, (2,2), strides=2))
model.add(MaxPooling2D(pool_size=(2,2))) 
model.add(Flatten()) #3차원을 reshape를 Faltten()으로!
model.add(Dense(100, activation='relu')) 
model.add(Dense(10, activation='softmax'))


# 3. 컴파일
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model3.load_weights('./save/weight_test02_3.h5')

#4. 평가, 예측
result3 = model3.evaluate(fm_x_test, fm_y_test, batch_size=32)
print("r3 loss : ", result3[0])
print("r3 acc : ", result3[1])

'''
    ValueError: Input 0 of layer sequential is incompatible with the layer: : 
expected min_ndim=4, found ndim=3. Full shape received: [None, 28, 28]    
'''