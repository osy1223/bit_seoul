from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
# flow :
# flow_from_directory : 
# 실제 데이터가 있는 곳을 알려주고, 이지를 불러오기
xy_train = train_datagen.flow_from_directory(
    './data/data1/train',
    target_size=(150, 150), #원래 이미지는 (150,150)이지만 (160, 160)도 가능
    batch_size=5, #이미지를 5장씩 잘라서 
    class_mode='binary'
    # x = 150,150,1(binayr)
    # y = ad, normal 라벨링 0,1
    # -> 한마디로, x랑 y가 같이 들어가 있는 상태
)

xy_test = test_datagen.flow_from_directory(
    './data/data1/test',
    target_size=(150, 150), #원래 이미지는 (150,150)이지만 (160, 160)도 가능
    batch_size=5, #이미지를 5장씩 잘라서 
    class_mode='binary'
)

# 모델 넣어줘야 합니다

model.fit_generator(
    xy_train,
    steps_per_epoch=100,
    epochs=20,
    valdation_data=xy_test, 
    validation_steps=4,
    save_to_dir='./data/data1_2/train' #변환된 사진 자동 저장
)