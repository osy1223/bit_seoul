import numpy as np
import os

#이미지 이름 저장
photo_idx = next(os.walk("./TeamProject/data/photo/camel/"))[2]
sketch_idx = next(os.walk("./TeamProject/data/sketch/camel/"))[2]

sketch = np.zeros((len(sketch_idx),256,256,3),dtype=np.uint8) 
photo = np.zeros((len(sketch_idx),256,256,3),dtype=np.uint8)

from tqdm import tqdm_notebook
from keras.preprocessing.image import img_to_array, load_img
from skimage.transform import resize #사이즈 조절

for i, idx in tqdm_notebook(enumerate(sketch_idx),total=len(sketch_idx)):
    print("i : ",i)
    print("idx : ",idx)
    img = load_img("./TeamProject/data/sketch/camel/"+idx)
    img = img_to_array(img)
    sketch[i] = img

for i, idx in tqdm_notebook(enumerate(sketch_idx),total=len(sketch_idx)):
    print("i : ",i)
    print("idx : ",idx)
    idx = idx.split("-")[0]
    img = load_img("./TeamProject/data/photo/camel/"+ idx+".jpg")
    img = img_to_array(img)
    photo[i] = img

print(sketch.shape) #(690, 256, 256, 3)
print(photo.shape) #(690, 256, 256, 3)

np.save('./TeamProject/npy/camel_sketch.npy', arr=sketch)
np.save("./TeamProject/npy/camel_photo.npy", arr=photo)
print('save npy')



from os import listdir
from numpy import asarray
from numpy import vstack
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import savez_compressed

# def load_images(path, size=(256,256)):
#     sketch_list, photo_list = list(), list()
#     for filename in listdir(path):
#         pixels = load_img(path + filename, target_size=size)
#         pixels = img_to_array(pixels)
#         sketch, photo = pixels[:, :256], pixels[:, 256:]
#         sketch_list.append(sketch)
#         photo_list.append(photo)
#     return [asarray(sketch_list), asarray(photo_list)]

# path1 = "./TeamProject/data/photo/camel/"+ idx+".jpg"
# path2 = "./TeamProject/data/sketch/camel/"+ idx+".jpg"

# [sketch_list, photo_list] = load_images[(path1, path2)]
# print('Loaded :', sketch_list.shape, photo_list.shape)

# filename = 'camel.npy'
# savez_compressed(filename, sketch_list, photo_list)
# print('Saved dataset:', filename)
