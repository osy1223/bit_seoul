import numpy as np
from sklearn.datasets import load_boston
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
datasets = load_boston()

x = datasets.data
y = datasets.target

print("x.shape:", x.shape) #x.shape: (506, 13)


# 1.1 데이터 전처리 
# OneHotEncoding

# append (데이터 합치기)

# reshape

# 1.3 Scaler -> 여기서의 스케일러는 PCA를 위한 스케일러
scaler = StandardScaler()
scaler.fit(x) # fit하고
x= scaler.transform(x) # 사용할 수 있게 바꿔서 저장하자

# PCA
pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) +1 #d : 우리가 필요한 n_components의 개수
print("n_components:",d) # n_components: 2 -> 스케일러 적용후 n_components: 9


pca = PCA(n_components=d)
x2d = pca.fit_transform((x))
print(x2d.shape) #(506, 9)


# pca_EVR = pca.explained_variance_ratio_
# print(sum(pca_EVR)) #0.9688751429772723


# 1.2 train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8)
print("x.split shape:", y_train.shape, x_test.shape) #(404,) (102, 13)
print("y.split shape:", x_train.shape, x_test.shape) #(404, 13) (102, 13)


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
    patience=10, 
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

cumsum >= 0.95
n_components: 9
RMSE : 5.231020009862587
R2 :  0.7033376896777574

cumsum >= 1
n_components: 1
RMSE : 5.010793605675593
R2 :  0.6653132812441837
'''