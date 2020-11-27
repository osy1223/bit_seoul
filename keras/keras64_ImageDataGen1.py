# 넘파이 저장
# 남자 여자를 
# fit_generater로 코딩

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout


np.random.seed(33) #33이라는 랜덤 난수를 사용하겠다.

# 이미지에 대한 생성 옵션 정하기 (트레인 이미지)
train_datagen = ImageDataGenerator(
    rescale= 1./255, # 정규화
    horizontal_flip= True, #True로 설정할 경우, 50% 확률로 이미지를 수평으로 뒤집습니다.
    vertical_flip= True,

    width_shift_range= 0.1,
    height_shift_range= 0.1,
    # 그림을 수평 또는 수직으로 랜덤하게 평행 이동시키는 범위 (원본 가로, 세로 길이에 대한 비율 값)

    rotation_range= 5, #이미지 회전 범위 (degrees)
    zoom_range= 1.2, #임의 확대/축소 범위
    shear_range= 0.7, #좌표 이동? 좌표 고정? 임의 전단 변환 (shearing transformation) 범위
    fill_mode='nearest' #이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식
    #nearest : 주변과 비슷하게 채워준다
)

test_datagen = ImageDataGenerator(rescale=1./255) # (테스트 이미지)
# 기존 이미지 갖고 테스트하는거라, 반전 또는 이것저것 할 필요 없다


# 이미지 불러오는 작업
# flow 또는 flow_from_directory
# 실제 데이터가 있는 곳을 알려주고, 이지를 불러오기
xy_train = train_datagen.flow_from_directory(
    './data/data2',
    target_size=(300, 300), 
    batch_size=100, #이미지를 5장씩 잘라서 
    class_mode='binary'
    # x = 150,150,1(binayr)
    # y = ad, normal 라벨링 0,1
    # -> 한마디로, x랑 y가 같이 들어가 있는 상태
)

xy_test = test_datagen.flow_from_directory(
    './data/data2', 
    target_size=(300, 300), 
    batch_size=100, #이미지를 5장씩 잘라서 
    class_mode='binary'
)

'''
Found 1736 images belonging to 2 classes.
Found 1736 images belonging to 2 classes.
'''

print('================================')
print(type(xy_train))
print('xy_train[0]:',xy_train[0])



# print(xy_train[0].shape) -> Error
# AttributeError: 'tuple' object has no attribute 'shape'
print('================================')
print('xy_train[0][0]):',xy_train[0][0])
print(type(xy_train[0][0])) #<class 'numpy.ndarray'>
print(xy_train[0][0].shape) #(100, 300, 300, 3)
print(xy_train[0][1].shape) #(100,)
# print(xy_train[1][1].shape) 
# print(xy_train[1][0].shape) 
# 총 160장의 이미지에서 bath_size로 잘리고

print(len(xy_train))  #18
# batch_size=100 이면 (100, 300, 300, 3) // len = 18

print('================================')
print('xy_train[0][0][0]:',xy_train[0][0][0]) # 첫번째 확인
print(xy_train[0][1][:10]) # y값 10개

'''
# 넘파이로 바꿔서 작업하세요~
np.save('./data/keras63_train_x.npy', arr=xy_train[0][0])
np.save('./data/keras63_train_y.npy', arr=xy_train[0][1])
np.save('./data/keras63_test_x.npy', arr=xy_test[0][0])
np.save('./data/keras63_test_y.npy', arr=xy_test[0][1])
# batch_size=200으로 train,test 다 변경해서 다 저장

# 모델 넣어줘야 합니다
model = Sequential()
model.add(Conv2D(10, (3,3), input_shape=(300,300,3)))
model.add(Conv2D(20, (2,2)))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Flatten())
model.add(Dense(50))
model.add(Dense(1, activation='sigmoid'))


hist = model.fit_generator(
    xy_train,
    steps_per_epoch=100,
    epochs=20,
    valdation_data=xy_test, 
    validation_steps=4,
    save_to_dir='./data/data1_2/train' #변환된 사진 자동 저장
)

# 평가, 예측
loss,acc = model.evaluate(xy_test, batch_size=32)

acc = hist.history['acc']
val_acc = hist.history['val_acc']
y_vloss = hist.history['val_loss']
y_loss = hist.history['loss']

print('acc:',acc)
print('val_acc:',val_acc)
print('y_vloss:',y_vloss)
print('y_loss:',y_loss)


#시각화 
plt.figure(figsize=(10,6)) #단위 무엇인지 찾아볼것!

plt.subplot(2,1,1) #(2행 1열에서 1번째 그림)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss') 
# x값과, y값이 들어가야 합니다
# plt.plot([x축 데이터], [y축 데이터]) 
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('val_loss')
plt.xlabel('epoch')
plt.legend(loc='upper right') #우측 상단에 legend(label 2개 loss랑 val_loss) 표시

plt.subplot(2,1,2) #(2행 1열에서 2번째 그림)
plt.plot(hist.history['acc'], marker='.', c='red')
plt.plot(hist.history['val_acc'], marker='.', c='blue')
plt.grid()

plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()
'''