# 66_1 copy
# lr 하고 마구마구 더 넣아라~

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, Adadelta

from tensorflow.keras.activations import relu, elu, selu, softmax, sigmoid
from tensorflow.keras.layers import ReLU, ELU, LeakyReLU, Softmax
from tensorflow.keras.layers import Activation

# 데이터 전처리 1. x,y train, test split
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 전처리 1. OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.


# 모델
def build_model(drop=0.5, optimizer=Adam, learning_rate=0.001):
    inputs = Input(shape=(28*28,), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, name='hidden2')(x)
    x = Activation('relu')(x)
    x = Dense(128, name='hidden3')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=optimizer(lr=learning_rate), 
        metrics=['acc'],
        loss='categorical_crossentropy')
    return model

def create_hyperparameters():
    batches = [100]
    optimizers = [Adam, Adadelta]
    learning_rate = [0.001, 0.1]
    activation = [ReLU, ELU, LeakyReLU]
    dropout = [0.1]
    epochs = [10]
    return{'batch_size':batches, 
            'optimizer':optimizers,
            'learning_rate':learning_rate,
            'drop':dropout,
            'epochs':epochs}
hyperparameters = create_hyperparameters()

# 우리가 만든 케라스 모델을 싸이킷런에 넣을수 있게 wrapping!
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=build_model, verbose=1)

# wrapping된 모델을 이용해서 싸이킷런의 RandomizedSearchCV 이용!
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# search = GridSearchCV(model, hyperparameters, cv=3)
search = RandomizedSearchCV(model, hyperparameters, cv=3)

es = EarlyStopping(
    monitor='loss',
    patience=10,
    mode='auto'
)

search.fit(x_train, y_train,
    epochs=100,
    callbacks=[es])

# 평가
acc = search.score(x_test, y_test)
print('최적의 파라미터 :', search.best_params_)
print('최종 스코어:', acc)


'''
최적의 파라미터 : {'optimizer': <class 'tensorflow.python.keras.optimizer_v2.adam.Adam'>, 'learning_rate': 
0.001, 'epochs': 10, 'drop': 0.1, 'batch_size': 100}
최종 스코어: 0.9836999773979187

마지막 모델로 돌렸을 때,
최적의 파라미터 : {'optimizer': <class 'tensorflow.python.keras.optimizer_v2.adam.Adam'>, 'learning_rate': 0.001, 'epochs': 10, 'drop': 0.1, 'batch_size': 100}
최종 스코어: 0.9869999885559082
'''




