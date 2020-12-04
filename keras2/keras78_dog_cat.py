from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# 이미지 불러오기 load_img
img_dog = load_img('./data/dog_cat/dog.jpg', target_size=(224, 224))
# plt.imshow(img_dog)
# plt.show()
img_cat = load_img('./data/dog_cat/cat.jpg', target_size=(224, 224))
img_suit = load_img('./data/dog_cat/suit.jpg', target_size=(224, 224))

# vgg16 디폴트값 : [(None, 224, 224, 3)]  

#이미지 수치화 img_to_array
arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_suit = img_to_array(img_suit)

# print(arr_dog)
# print(type(arr_dog)) #<class 'numpy.ndarray'>
# print(arr_dog.shape) #(512, 512, 3)
# print(arr_cat.shape) #(700, 700, 3)


# RGB -> BGR (텐서플로우는 이미지 shape 바꿔줘야합니다)
from tensorflow.keras.applications.vgg16 import preprocess_input
arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_suit = preprocess_input(arr_suit)


# print(arr_dog)
# print(arr_dog.shape) #(512, 512, 3) 3이 바뀐겁니다. 
# print(arr_cat.shape) #(700, 700, 3)

arr_input = np.stack([arr_dog, arr_cat, arr_suit])
print(arr_input.shape)


# 2. 모델 구성
model = VGG16()
probs = model.predict(arr_input)
print(probs)
print('probs.shape:', probs.shape)

# 이미지 결과 확인 (복호화)
from tensorflow.keras.applications.vgg16 import decode_predictions

results = decode_predictions(probs)
print('---------------------------')
print('result[0] :', results[0])
print('---------------------------')
print('result[1] :', results[1])
print('---------------------------')
print('result[2] :', results[2])

'''
---------------------------
result[0] : [('n02113023', 'Pembroke', 0.9494068), ('n02113186', 'Cardigan', 0.042273775), ('n02099601', 'golden_retriever', 0.0017585789), ('n02112018', 'Pomeranian', 0.0016593949), ('n02085620', 'Chihuahua', 0.00053557317)]
---------------------------
result[1] : [('n02085936', 'Maltese_dog', 0.67439586), ('n02123394', 'Persian_cat', 0.058075406), ('n02086079', 'Pekinese', 0.040431425), ('n02883205', 'bow_tie', 0.029396538), ('n02328150', 'Angora', 0.020199958)]
---------------------------
result[2] : [('n04350905', 'suit', 0.6809431), ('n03770439', 'miniskirt', 0.083943464), ('n03680355', 'Loafer', 0.082788534), ('n04479046', 'trench_coat', 0.022563169), ('n02883205', 'bow_tie', 0.013991045)]
'''