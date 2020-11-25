# pca로 축소해서 모델을 완성하시오
# 1. 0.95 이상
# 2. 1 이상
# mnist dnn loss와 acc를 비교

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard 
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) #(60000,) (10000,)

# 1.1 데이터 전처리 

# append (데이터 합치기)
x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)
print("x.shape:", x.shape) #x.shape: (70000, 28, 28)

# reshape
x = x.reshape(70000, 28*28)
print(x.shape) #(70000, 784)

# PCA
pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) +1 #d : 우리가 필요한 n_components의 개수
print("n_components:",d) #n_components: 713


pca = PCA(n_components=713)
x2d = pca.fit_transform((x))
print(x2d.shape) # (70000, 713)


pca_EVR = pca.explained_variance_ratio_
print(sum(pca_EVR))  #1.0000000000000024

# OneHotEncoding
y = to_categorical(y)

# 1.2 train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=60000, test_size=10000)
print("split shape:",x_train.shape, x_test.shape)

# 이미지를 DNN 할 때에는 reshape하고 scaler 하자
# 1.3 Scaler
scaler = MinMaxScaler()
scaler.fit(x_train) # fit하고
x_train = scaler.transform(x_train) # 사용할 수 있게 바꿔서 저장하자
x_test = scaler.transform(x_test) # 사용할 수 있게 바꿔서 저장하자

# 2. 모델
model = Sequential()
model.add(Dense(10, input_shape=(x_train.shape[1],))) 
model.add(Dense(20)) 
model.add(Dense(50))
model.add(Dense(40)) 
model.add(Dense(100, activation='relu')) 
model.add(Dense(10, activation='softmax')) #분류한 값의 총 합은 1

# 3. 컴파일, 훈련
model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['acc'])

es = EarlyStopping(
    monitor='loss', 
    patience=20, 
    mode='auto',
    verbose=2)

model.fit(x_train,y_train, 
        epochs=1000, 
        batch_size=128, 
        verbose=1, 
        validation_split=0.2, 
        callbacks=[es])

# 4. 평가, 예측
loss,acc = model.evaluate(x_test, y_test, batch_size=128)
print("loss : ",loss)
print("acc : ",acc)


'''
mnist dnn loss와 acc
input_shape=(784,)
loss :  0.24711093306541443
acc :  0.9215999841690063


0.95 이상일 때,
우리가 필요한 n_components의 개수 : 154
fit_transform : (70000, 154)
explained_variance_ratio_ : 0.9499857838055975
loss :  0.3831883668899536
acc :  0.9509999752044678

1 이상 일 때,
우리가 필요한 n_components의 개수 : 713
fit_transform : (70000, 713)
explained_variance_ratio_ : 1.0000000000000024
loss :  0.40088239312171936
acc :  0.9509000182151794
'''