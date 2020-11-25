import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard 

# 1. 데이터
datasets = load_breast_cancer()
# print(datasets) #data, target

x = datasets.data
y = datasets.target

print('x.shape:', x.shape) #x.shape: (569, 30)

# 2. 데이터 전처리
# OneHotEncoding -> 분류 데이터니깐 합시다!
y = to_categorical(y)

# reshape

# scaler -> PCA를 위한 스케일러
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# PCA
cumsum_standard = 1
pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= cumsum_standard) +1
print('n_components:', d) #n_components: 10

pca = PCA(n_components=d)
x = pca.fit_transform(x)
print('after pca x.shape:', x.shape) #after pca x.shape: (569, 10)

#  train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8
)
print("split x_train.shape:", x_train.shape, x_test.shape) #(455, 10) (114, 10)

# 2. 모델
model = Sequential()
model.add(Dense(400, input_shape=(x_train.shape[1],)))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(2, activation='sigmoid')) #이진분류

# 3. 컴파일, 훈련
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping # 조기 종료
early_stopping = EarlyStopping(
    monitor='accuracy',
    patience=50,
    mode='auto',
    verbose=2)

history = model.fit(x_train, y_train,
    epochs=500,
    verbose=0,
    validation_split=0.2,
    callbacks=[early_stopping], 
    batch_size=128)

# 4. 평가 및 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size=128)
print("loss: ", loss)
print("accuracy: ", accuracy)


y_predict = model.predict(x_test) # 평가 데이터 다시 넣어 예측값 만들기
y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)
print("y_test:\n", y_test)
print("y_predict:\n", y_predict)


# 사용자정의 RMSE 함수
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE:", RMSE(y_test, y_predict))

# 사용자정의 R2 함수
# 사이킷런의 metrics에서 r2_score를 불러온다
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2:", r2)


#모델 시각화
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
acc_ax.plot(history.history['accuracy'], 'b', label='train accuracy')
acc_ax.plot(history.history['val_accuracy'], 'g', label='val accuracy')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax. set_ylabel('accuracy')
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
 
plt.show()
model.summary()

'''
cancer DNN
loss :  0.09477243572473526
acc :  0.9590643048286438

cumsum_standard = 0.95일 때,
n_components: 10
RMSE: 0.24779731389167603
R2: 0.7429951690821256

cumsum_standard = 1일 때,
n_components: 30
RMSE: 0.24779731389167603
R2: 0.7494505494505495
'''