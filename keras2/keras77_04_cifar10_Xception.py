# 실습
# 가장 좋은 놈 어떤건지 결과치 비교용
# 기본튠 + 전이학습 9개 모델 비교

# 9개의 전이학습 모델들은
# Flatten() 다음에는 모두 똑같은 레이어로 구성

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split 
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import NASNetLarge

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) 

# CNN을 위한 reshape
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3])
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_train.shape[3])
print("reshape x:", x_train.shape, x_test.shape)

# 2. 모델 
model1 = NASNetLarge(
    weights='imagenet', 
    include_top=False, 
    input_shape=(32,32,3)
)

model1.trainable = False

model = Sequential()
model.add(model1)
model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
# model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam',
    metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=6
)

modelpath = './model/cancer-{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(
    filepath=modelpath,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss',
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    patience=3,
    factor=0.5,
    verbose=1
)

model.fit(x_train, y_train, 
    epochs=30, 
    batch_size=32, 
    verbose=1, 
    validation_split=0.5,
    callbacks=[es, cp]
)

# 평가, 예측
loss,acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ",loss)
print("acc : ",acc)
print(model1.name)

'''
Xception 모델은 이미지 큰걸로만 가능!
ValueError: Input size must be at least 71x71; got `input_shape=(32, 32, 3)`

InceptionResNetV2 모델은 이미지 큰걸로만 가능!
ValueError: Input size must be at least 75x75;

NASNetLarge 모델은 
ValueError: When setting `include_top=True` and loading `imagenet` weights, `input_shape` should be (331, 331, 3).
'''