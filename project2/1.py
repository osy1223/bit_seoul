import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
from conda.models import channel


plt.style.use('dark_background')

img_ori = cv2.imread('./project2/img/1.jpg')
height, width, channer = img_ori.shape
print(img_ori.shape) #(626, 940, 3)

#시각화 
plt.figure(figsize=(12,10))
plt.imshow(img_ori, cmap='gray')
plt.show()

gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(12,10))
plt.imshow(gray, cmap='gray')
plt.show()

# 노이즈를 줄이기 위한 함수
img_blurred = cv2.GaussianBlur(gray, ksize=(5,5), sigmaX=0)


# threshold only vs blur and threshold 비교
#threshold only
img_thresh = cv2.adaptiveThreshold(
    gray,
    maxValue=255.0,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=19,
    C=9
)


# 이미지 구분하기 쉽게 (blur+thresholding)
# blur and threshold
img_blurred = cv2.GaussianBlur(gray, ksize=(5,5), sigmaX=0)

img_blur_thresh = cv2.adaptiveThreshold(
    img_blurred,
    maxValue=255.0,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=19,
    C=9
)


#시각화 blur and threshold 비교
plt.figure(figsize=(12,10))
plt.subplot(1,2,1)
plt.title('Threshold only')
plt.imshow(img_thresh, cmap='gray')


plt.subplot(1,2,2)
plt.title('Blur + Threshold')
plt.imshow(img_thresh, cmap='gray')
plt.imshow(img_blur_thresh, cmap='gray')
plt.show()


# 윤곽선 찾기
contours = cv2.findContours(
    img_thresh,
    mode=cv2.RETR_LIST,
    method=cv2.CHAIN_APPROX_SIMPLE
)

temp_result = np.zeros((height, width, channel), dtype=np.uint8)

cv2.drawContours(temp_result, 
    contours=contours, 
    contourIdx=-1,
    color=(255, 255, 255))

plt.figure(figsize=(12,10))
plt.imshow(temp_result)
plt.show()


contours_dict = []
# 컨투어의 사각형 범위 찾아내기
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(temp_result, 
                pt1=(x,y), 
                pt2=(x+w, y+h), 
                color=(255,255,255),
                thickness=2)

# contours_dict에 저장
contours_dict.append({
    'contour':contour,
    'x':x,
    'y':y,
    'w':w,
    'h':h,
    'cx':x+(w/2),
    'cy':y+(h/2)
})


MIN_AREA =80
MIN_WIDTH, MIN_HEIGHT =2,8
MIN_RATIO, MAX_RATIO = 0.25, 1.0

possible_countours = []

cnt = 0
for d in contours_dict:
    area = d['w']*d['h']
    ratio = d['w']/d['h']

    if area > MIN_AREA \
    and d['w']>MIN_WIDTH and d['h']>MIN_HEIGHT\
    and MIN_RATIO<ratio<MAX_RATIO:
        d['idx']=cnt
        cnt +=1
        possible_countours.append(d)

temp_result=np.zeros