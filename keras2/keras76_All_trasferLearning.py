# 모델별로 가장 순수했을때의, 파라미터의 갯수와 가중치 수를 정리하시오

from tensorflow.keras.applications import VGG16, VGG19, Xception
from tensorflow.keras.applications import ResNet101, ResNet101V2
from tensorflow.keras.applications import ResNet152, ResNet152V2
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import NASNetLarge, NASNetMobile

from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout, Activation
from tensorflow.keras.models import Sequential

model = NASNetMobile() 
model.trainable = True 
model.summary()
print('동결하기 전 훈련되는 가중치의 수 :', len(model.trainable_weights))
print(model.name)

# VGG16 모델은 Total params: 138,357,544 // 동결하기 전 훈련되는 가중치의 수 : 32
# VGG19 모델은 Total params: 143,667,240 // 동결하기 전 훈련되는 가중치의 수 : 38
# Xception 모델은 Total params: 22,910,480 // 동결하기 전 훈련되는 가중치의 수 : 156
# ResNet101 모델은 Total params: 44,707,176 // 동결하기 전 훈련되는 가중치의 수 : 418
# ResNet101V2 모델은 Total params: 44,675,560 // 동결하기 전 훈련되는 가중치의 수 : 344
# ResNet152 모델은 Total params: 60,419,944 // 동결하기 전 훈련되는 가중치의 수 : 622
# ResNet152V2 모델은 Total params: 60,380,648 // 동결하기 전 훈련되는 가중치의 수 : 514
# InceptionV3 모델은 Total params: 23,851,784 // 동결하기 전 훈련되는 가중치의 수 : 190
# InceptionResNetV2 모델은 Total params: 55,873,736 // 동결하기 전 훈련되는 가중치의 수 : 490
# MobileNet 모델은 Total params: 4,253,864 // 동결하기 전 훈련되는 가중치의 수 : 83
# MobileNetV2 모델은 Total params: 3,538,984 // 동결하기 전 훈련되는 가중치의 수 : 158
# DenseNet121 모델은 Total params: 8,062,504 // 동결하기 전 훈련되는 가중치의 수 : 364
# DenseNet169 모델은 Total params: 14,307,880 // 동결하기 전 훈련되는 가중치의 수 : 508
# DenseNet201 모델은 Total params: 20,242,984 // 동결하기 전 훈련되는 가중치의 수 : 604
# NASNetLarge 모델은 Total params: 88,949,818 // 동결하기 전 훈련되는 가중치의 수 : 1018
# NASNetMobile 모델은 Total params: 5,326,716 // 동결하기 전 훈련되는 가중치의 수 : 742








