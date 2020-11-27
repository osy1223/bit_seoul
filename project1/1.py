import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
import pandas as pd

from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from keras.layers import Input, Activation, Conv2D, Flatten, Dense, MaxPooling2D, Dropout, Add, LeakyReLU
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

import datetime
import os, sys, glob

# 데이터 로드 csv 파일
labels = pd.read_csv('./project1/train_test_files/All_labels.txt',
    sep="",
    header=None
)
print(labels)
labels.head()
labels.describe()

print(labels.shape) #(5500, 2)

# 데이터 변경 np 파일
labels_np = labels.values
print(labels_np[:5])
'''
[['CF437.jpg' 2.8833330000000004]
 ['AM1384.jpg' 2.4666669999999997]
 ['AM1234.jpg' 2.15]
 ['AM1774.jpg' 3.75]
 ['CF215.jpg' 3.0333330000000003]]
'''

# img를 담을 imgs
imgs = np.empty((len(labels_np), 350, 350, 3), dtype=np.uint8)

for i, (img_filename, rating) in enumerate(labels_np):
    img = cv2.imread(os.path.join('Images', img_filename))
    print(type(img))
    if img.shape[0] != 350 or img.shape[1] != 350:
        print(img_filename)

    imgs[i] = img

'''
# train_test_split
x_train, x_val, y_train, y_val = train_test_split(
    imgs, labels_np[: , 1], test_size=0.1)

np.save('x_train.npy', x_train)
np.save('x_val.npy', x_val)
np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)

print(x_train.shape, x_val.shape)
print(y_train.shape, y_val.shape)

# 시각화
plt.figure(figsize=(16,6))
for i, img in enumerate(x_train[:10]):
    plt.subplot(2, 5, i+1)
    plt.axis('off')
    plt.title('%.2f' %y_train[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow()

# 데이터셋 인위적으로 늘리기
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.2,
    zoom_range=0.4,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)


# 정규화
train_generator = train_datagen.flow(
    x = x_train, y=y_train,
    batch_size=32,
    shuffle=True
)

val_generator = val_datagen.flow(
    x=x_val, y=y_val,
    batch_size=32,
    shuffle=False
)

augs = train_generator.__getitem__(8)

plt.figure(figsize=(16,8))
for i, img in enumerate(augs[0]):
    plt.subplot(4,8,i+1)
    plt.title('%.2f' %augs[1][i])
    plt.axis('off')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img.squeeze())


# 모델링
inputs = Input(shape=(350,350,3))

net = Conv2D(32, kernel_size=3, strides=1, padding='same')(inputs)
net = LeakyReLU()(net)
net = MaxPooling2D(pool_size=2)(net)

net = Conv2D(32, kernal_size=3, strides=1, padding='same')(net)
net = LeakyReLU()(net)
net = MaxPooling2D(pool_size=2)(net)

net = Conv2D(32, kernal_size=3, strides=1, padding='same')(net)
net = LeakyReLU()(net)
net = MaxPooling2D(pool_size=2)(net)

net = Conv2D(32, kernal_size=3, strides=1, padding='same')(net)
net = LeakyReLU()(net)
net = MaxPooling2D(pool_size=2)(net)

net = Conv2D(32, kernal_size=3, strides=1, padding='same')(net)
net = LeakyReLU()(net)
net = MaxPooling2D(pool_size=2)(net)

net = Flatten()(net)

net = Dense(256)(net)
net = Activation('relu')(net)
net = Dense(128)(net)
net = Dense(1)(net)

outputs = Activation('linear')(net)

model = Model(inputs=inputs, outputs=outputs)


# 컴파일
model.compile(optimizer='adam', loss='mae')

model.summary()

# 훈련
start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

model.fit_generator(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[
        ModelCheckpoint('models/%s.h5' % (start_time), 
        monitor='val_loss',
        save_best_only=True, 
        mode='min', 
        verbose=1]
)

# 테스트

model = load_model('models/2020_11_26_21_00.h5')

val_data = val_generator.__getitem__(0)

preds = model.predict(val_data[0])

plt.figure(figsize=(16,8))
for i, img in enumerate(val_data[0]):
    plt.subplot(4, 8, i+1)
    plt.title('%.2f / %.2f' % (preds[i], val_data[1][i]))
    plt.axis('off')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img.squeeze())
    
'''