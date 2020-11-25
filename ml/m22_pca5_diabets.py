import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
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
datasets = load_diabetes()
# print(datasets) #data, target

x = datasets.data
y = datasets.target

print('x.shape:', x.shape) #x.shape: (442, 10)

# 2. 데이터 전처리
# OneHotEncoding

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
print('n_components:', d) #n_components: 8  

pca = PCA(n_components=d)
x = pca.fit_transform(x)
print('after pca x.shape:', x.shape) #after pca x.shape: (442, 8)

#  train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8
)
print("split x_train.shape:", x_train.shape, x_test.shape) #(353, 8) (89, 8)

# 2.모델
model = Sequential()
model.add(Dense(500, input_shape=(x_train.shape[1],)))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(
    loss='mse',
    optimizer='adam',
    metrics=['mae'])


from tensorflow.keras.callbacks import EarlyStopping # 조기 종료
early_stopping = EarlyStopping(
    monitor='loss',
    patience=100,
    mode='auto',
    verbose=2)

history = model.fit(x_train, y_train,
    epochs=10000,
    verbose=0,
    validation_split=0.25,
    callbacks=[early_stopping],
    batch_size=128)

# 4. 평가 및 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=128)
print("loss: ", loss) # 이건 기본으로 나오고
print("mae: ", mae)

y_predict = model.predict(x_test) # 평가 데이터 다시 넣어 예측값 만들기
print("y_predict:\n", y_test)
print("y_predict:\n", y_predict)
print("y_predict.shape:\n", y_predict.shape)

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

#모델 시각
fig, loss_ax = plt.subplots()
 
acc_ax = loss_ax.twinx()
 
loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
 
acc_ax.plot(history.history['mae'], 'b', label='train mae')
acc_ax.plot(history.history['val_mae'], 'g', label='val mae')
 
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax. set_ylabel('mae')
 
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
 
plt.show()
model.summary()

'''
diabets DNN
RMSE : 49.62671295349633
R2 :  0.5723361986594027

cumsum_standard = 0.95일 때,
n_components: 8
RMSE: 51.43341428179193
R2: 0.5175207352098945

cumsum_standard = 1일 때,
n_components: 10
RMSE: 57.4111596391751
R2: 0.3956118670050418

'''