
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Input, Dropout, LSTM

# 데이터 전처리 1. x,y train, test split
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, x_test.shape)  # (60000,28,28), (10000,28,28)
# print(y_train.shape, y_test.shape)  # (60000, )      (10000, )

# 데이터 전처리 1. OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_train.shape, y_test.shape) #(60000, 10) (10000, 10)  
# print(y_train[0]) #[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]

x_train = x_train.reshape(60000, 28, 28).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28).astype('float32')/255.
# print(x_train[0])


# 모델
def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(28, 28), name='input')
    x = LSTM(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')
    return model

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.4, 5)
    return{'batch_size':batches, 'optimizer':optimizers,
            'drop':dropout}
hyperparameters = create_hyperparameters()

# 우리가 만든 케라스 모델을 싸이킷런에 넣을수 있게 wrapping!
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=build_model, verbose=1)

# wrapping된 모델을 이용해서 싸이킷런의 GridSearchCV 이용!
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# search = GridSearchCV(model, hyperparameters, cv=3)
search = RandomizedSearchCV(model, hyperparameters, cv=3)
search.fit(x_train, y_train)

print('최적의 파라미터 :', search.best_params_)
acc = search.score(x_test, y_test)
print('최종 스코어:',acc)







