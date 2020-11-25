# Conv2D , Maxplling2D, Flatten

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(10,10,1)))  #9,9,10
#10:filters(output) // kernel_size(2,2):4이면, 2by2로 잘랐다 // strides 1이 디폴트
# 2/2 cnn 한번 통과하면 5/5가 4/4/10 으로 이미지 증폭 했다가, Dense로 줄여가며 output
model.add(Conv2D(5, (2,2), padding='same')) #9,9,5 padding이 same이면 input=output 이므로 9,9,10 -> 9,9,5
model.add(Conv2D(3, (3,3))) #7,7,3
model.add(Conv2D(7, (2,2))) #6,6,7
model.add(MaxPooling2D()) #3,3,7 (Maxpooling2D 한번 하면 반으로)
#현재 Input_shape 4차원 // Dense는 2차원이라 못 받아들입니다.
model.add(Flatten()) # 3*3*7=63 (노드의 개수 63개)
model.add(Dense(1)) # 최종 output이라 1

model.summary()

#input_shape=(rows, cols, channels) : input_shape=(10,10,1)
# 파라미터 계산법 
# (input(=channel) 1 x 커널사이즈(2*2) + bias 1 ) output 10 = 60
# (입력 10 x 2*2 +1)5 = 205
# (5 * (3*3) * +1)3 = 138

'''
Model: "sequential"
_________________________________________________________________        
Layer (type)                 Output Shape              Param #
=================================================================        
conv2d (Conv2D)              (None, 9, 9, 10)          50
_________________________________________________________________        
conv2d_1 (Conv2D)            (None, 9, 9, 5)           205
_________________________________________________________________        
conv2d_2 (Conv2D)            (None, 7, 7, 3)           138
_________________________________________________________________        
conv2d_3 (Conv2D)            (None, 6, 6, 7)           91
_________________________________________________________________        
max_pooling2d (MaxPooling2D) (None, 3, 3, 7)           0
_________________________________________________________________        
flatten (Flatten)            (None, 63)                0
_________________________________________________________________        
dense (Dense)                (None, 1)                 64
=================================================================        
Total params: 548
Trainable params: 548
Non-trainable params: 0
'''

'''
filters (output) (밖으로 나가는 노드의 개수)
kernel_size : 얼마만의 크기로 자를 것인가
strides : 몇칸씩 옮겨가면서 (1이 디폴트)
pading : valid가 디폴트 // same이면 padding을 적용시켜서 가장자리에 padding(통상적으로 0으로 패딩) (입력 shape가 그대로 다음에 전달)-> 데이터 손실을 막아줌
입력 모양 : batch_size, rows, cols, channels
input_shape=(rows, cols, channels)
shape : batch_size, rows, cols, channels
        (batch_size : 10장씩 작업하겠다.?
'''

'''
참고 LSTM
units (output) (밖으로 나가는 노드의 개수)
return_sequence
입력 모양 : batch_size, timesteps, feaure
input_shape = (timesteps, feature)
                timsteps :10일치씩 자르겠다.(시간의 간격의 규칙) ?
'''

'''
필터로 특징을 뽑아주는 컨볼루션(Convolution) 레이어
Conv2D(32, (5, 5), padding='valid', input_shape=(28, 28, 1), activation='relu')
-첫번째 인자 : 컨볼루션 필터의 수 입니다.
-두번째 인자 : 컨볼루션 커널의 (행, 열) 입니다.
-padding : 경계 처리 방법을 정의합니다.(‘valid’ : 유효한 영역만 출력이 됩니다. 따라서 출력 이미지 사이즈는 입력 사이즈보다 작습니다.
                                      ‘same’ : 출력 이미지 사이즈가 입력 이미지 사이즈와 동일합니다.)
-input_shape : 샘플 수를 제외한 입력 형태를 정의 합니다. 모델에서 첫 레이어일 때만 정의하면 됩니다.
(행, 열, 채널 수)로 정의합니다. 흑백영상인 경우에는 채널이 1이고, 컬러(RGB)영상인 경우에는 채널을 3으로 설정합니다.
-activation : 활성화 함수 설정합니다.
    ‘linear’ : 디폴트 값, 입력뉴런과 가중치로 계산된 결과값이 그대로 출력으로 나옵니다.
    ‘relu’ : rectifier 함수, 은익층에 주로 쓰입니다.
    ‘sigmoid’ : 시그모이드 함수, 이진 분류 문제에서 출력층에 주로 쓰입니다.
    ‘softmax’ : 소프트맥스 함수, 다중 클래스 분류 문제에서 출력층에 주로 쓰입니다.
'''
