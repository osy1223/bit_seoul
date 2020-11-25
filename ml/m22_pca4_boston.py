import numpy as np
from tensorflow.keras.datasets import boston_housing
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
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape) # (404, 13) (102, 13)
print(y_train.shape, y_test.shape) # (404,) (102,)


# 1.1 데이터 전처리 
# append (데이터 합치기)
x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)
print("x.shape:", x.shape) # x.shape: (506, 13)


# reshape


# PCA
pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) +1 #d : 우리가 필요한 n_components의 개수
print("n_components:",d) # n_components: 2// 1


pca = PCA(n_components=2)
x2d = pca.fit_transform((x))
print(x2d.shape) #(506, 2)


pca_EVR = pca.explained_variance_ratio_
print(sum(pca_EVR)) #0.9688751429772723

# OneHotEncoding
y = to_categorical(y)

# 1.2 train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=500)
print("split shape:", x_train.shape, x_test.shape) # 

# 1.3 Scaler
scaler = StandardScaler()
scaler.fit(x_train) # fit하고
x_train = scaler.transform(x_train) # 사용할 수 있게 바꿔서 저장하자
x_test = scaler.transform(x_test) # 사용할 수 있게 바꿔서 저장하자

# 2.모델

model = Sequential()
model.add(Dense(10, input_shape=(x_train.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1))

# 컴파일, 훈련
model.compile(
    loss='mse', 
    optimizer='adam', 
    metrics=['mse'])

es = EarlyStopping(
    monitor='loss', 
    patience=5, 
    mode='auto')

model.fit(x_train, y_train, 
    epochs=300, 
    batch_size=10, 
    verbose=1, 
    validation_split=0.2, 
    callbacks=[es])


# 평가, 예측
y_predict = model.predict(x_test)

# RMSE, R2
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE :", RMSE(y_test,y_predict))

# R2 함수
from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)
print("R2 : ",r2)


'''
boston DNN
RMSE : 4.639773992883862
R2 :  0.7306575547275955


'''