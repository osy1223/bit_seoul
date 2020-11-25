import numpy as np
from tensorflow.keras.datasets import cifar100
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard 
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, x_test.shape) # (50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape) # (50000, 1) (10000, 1)

# 1.1 데이터 전처리 
# append (데이터 합치기)
x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)
print("x.shape:", x.shape) #x.shape:(60000, 32, 32, 3)


# reshape
x = x.reshape(60000, 32*32*3)
print(x.shape) # (60000, 3072)


# PCA
pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) +1 #d : 우리가 필요한 n_components의 개수
print("n_components:",d) # n_components: 202


pca = PCA(n_components=202)
x2d = pca.fit_transform((x))
print(x2d.shape) #(60000, 202)


pca_EVR = pca.explained_variance_ratio_
print(sum(pca_EVR))  # 1.0000000000000022

# OneHotEncoding
y = to_categorical(y)

# 1.2 train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=500)
print("split shape:", x_train.shape, x_test.shape) # (500, 3072) (59500, 3072)

# 이미지를 DNN 할 때에는 reshape하고 scaler 하자
# 1.3 Scaler
scaler = StandardScaler()
scaler.fit(x_train) # fit하고
x_train = scaler.transform(x_train) # 사용할 수 있게 바꿔서 저장하자
x_test = scaler.transform(x_test) # 사용할 수 있게 바꿔서 저장하자

# 2. 모델
model = Sequential()
model.add(Dense(10, input_shape=(x_train.shape[1],))) 
model.add(Dense(500, activation='relu')) 
model.add(Dropout(0.2)) 
model.add(Dense(100))
model.add(Dropout(0.2)) 
model.add(Dense(300)) 
model.add(Dense(100))
model.add(Dense(50)) 
model.add(Dense(3)) 
model.add(Dense(100, activation='softmax'))

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
cifar100 dnn
loss :  3.6389894485473633
acc :  0.14429999887943268

cumsum >= 0.95
n_components : 202
loss :  21.481054306030273
acc :  0.01484033651649952


cumsum >= 1
n_components: 3072
loss :  23.00354766845703
acc :  0.016857143491506577
'''