import numpy as np

dataset = np.array(range(1, 101))
size = 5

# 실습 : 모델 구성
# train, test 분리
# early_stopping

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size +1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset]) #subset
    print(type(aaa))
    return np.array(aaa)

datasets = split_x(dataset, size)
print(datasets)
'''
[[  1   2   3   4   5]
 [  2   3   4   5   6]
 [  3   4   5   6   7]
 [  4   5   6   7   8]
 [  5   6   7   8   9]
 [  6   7   8   9  10]
 [  7   8   9  10  11]
 [  8   9  10  11  12]
 [  9  10  11  12  13]
 [ 10  11  12  13  14]
 [ 11  12  13  14  15]
 [ 12  13  14  15  16]
 [ 13  14  15  16  17]
 [ 14  15  16  17  18]
 [ 15  16  17  18  19]
 [ 16  17  18  19  20]
 [ 17  18  19  20  21]
 [ 18  19  20  21  22]
 [ 19  20  21  22  23]
 [ 20  21  22  23  24]
 [ 21  22  23  24  25]
 [ 22  23  24  25  26]
 [ 23  24  25  26  27]
 [ 24  25  26  27  28]
 [ 25  26  27  28  29]
 [ 26  27  28  29  30]
 [ 27  28  29  30  31]
 [ 28  29  30  31  32]
 [ 29  30  31  32  33]
 [ 30  31  32  33  34]
 [ 31  32  33  34  35]
 [ 32  33  34  35  36]
 [ 33  34  35  36  37]
 [ 34  35  36  37  38]
 [ 35  36  37  38  39]
 [ 36  37  38  39  40]
 [ 37  38  39  40  41]
 [ 38  39  40  41  42]
 [ 39  40  41  42  43]
 [ 40  41  42  43  44]
 [ 41  42  43  44  45]
 [ 42  43  44  45  46]
 [ 43  44  45  46  47]
 [ 44  45  46  47  48]
 [ 45  46  47  48  49]
 [ 46  47  48  49  50]
 [ 47  48  49  50  51]
 [ 48  49  50  51  52]
 [ 49  50  51  52  53]
 [ 50  51  52  53  54]
 [ 51  52  53  54  55]
 [ 52  53  54  55  56]
 [ 53  54  55  56  57]
 [ 54  55  56  57  58]
 [ 55  56  57  58  59]
 [ 56  57  58  59  60]
 [ 57  58  59  60  61]
 [ 58  59  60  61  62]
 [ 59  60  61  62  63]
 [ 60  61  62  63  64]
 [ 61  62  63  64  65]
 [ 62  63  64  65  66]
 [ 63  64  65  66  67]
 [ 64  65  66  67  68]
 [ 65  66  67  68  69]
 [ 66  67  68  69  70]
 [ 67  68  69  70  71]
 [ 68  69  70  71  72]
 [ 69  70  71  72  73]
 [ 70  71  72  73  74]
 [ 71  72  73  74  75]
 [ 72  73  74  75  76]
 [ 73  74  75  76  77]
 [ 74  75  76  77  78]
 [ 75  76  77  78  79]
 [ 76  77  78  79  80]
 [ 77  78  79  80  81]
 [ 78  79  80  81  82]
 [ 79  80  81  82  83]
 [ 80  81  82  83  84]
 [ 81  82  83  84  85]
 [ 82  83  84  85  86]
 [ 83  84  85  86  87]
 [ 84  85  86  87  88]
 [ 85  86  87  88  89]
 [ 86  87  88  89  90]
 [ 87  88  89  90  91]
 [ 88  89  90  91  92]
 [ 89  90  91  92  93]
 [ 90  91  92  93  94]
 [ 91  92  93  94  95]
 [ 92  93  94  95  96]
 [ 93  94  95  96  97]
 [ 94  95  96  97  98]
 [ 95  96  97  98  99]
 [ 96  97  98  99 100]]
'''

x = datasets[:, 0:4]
y = datasets[:, 4]

x = np.reshape(x, (x.shape[0], x.shape[1], 1)) #3차원
# x = np.reshape(x, (x.shape[0], x.shape[1], 1, 1)) #4차원 
print(x.shape) # (96, 4, 1) 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7
)

#96*(train_size가 0.7이라서 = 67)
print(x_train.shape) #(67, 4, 1)

# LSTM 함수형 모델 구성
from tensorflow.keras.models import Model , Sequential
from tensorflow.keras.layers import Dense, LSTM, Input

input1 = Input(shape=(4, 1)) 
dense1 = LSTM(100, activation='relu')(input1)
dense2 = Dense(200, activation='relu')(dense1)
dense3 = Dense(300, activation='relu')(dense2)
output = Dense(1)(dense3)

model = Model(inputs=input1, outputs=output)

model.summary()

# 컴파일, 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x,y, epochs=100, batch_size=1)

# 평가, 예측
x_pred = np.array([97,98,99,100])

x_pred = x_pred.reshape(1,4,1)

y_predict = model.predict(x_pred)
print("y_pred :", y_predict)

loss, mse = model.evaluate(x_test, y_test)
print("loss :", loss)
print("mse :", mse)

'''
y_pred : [[100.81893]]

loss : 0.003341533476486802
mse : 0.003341533476486802
'''