import numpy as np
import cv2
from matplotlib import pyplot as plt


# cv2 이용
# 이미지 읽기
img = cv2.imread('./image/test.jpg', 1)

# 이미지 화면에 표시
cv2.imshow('Test Image:', img)

k = cv2.waitKey(0)

# 이미지 윈도우 삭제
cv2.destroyAllWindows()

# 이미지 다른 파일로 저장
cv2.imwrite('./image/after_test.png', img)


# Matplotlib 이용
img = cv2.imread('./image/test.jpg',0)
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])
plt.show()

