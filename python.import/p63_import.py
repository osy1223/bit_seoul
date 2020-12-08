

from test_1208 import p62_import
p62_import.sum2()
# 파일 불러온거고 
print('-------------------')

from test_1208.p62_import import sum2
sum2()
# 함수만 불러온거고

'''
이 파일은 아나콘다 폴더에 들어있을 것이다!
작업그룹 임포트 썸탄다!!
-------------------
작업그룹 임포트 썸탄다!
'''