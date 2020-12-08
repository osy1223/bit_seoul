import p11_car
import p12_tv

'''
import했을땐, 파일명 출력
car.py의  module 이름은 p11_car
tv.py의  module 이름은 p12_tv
'''

print('----------------------')
print('do.py의 module 이름은', __name__)
print('----------------------')
'''
do.py의 module 이름은 __main__
'''

p11_car.drive()
p12_tv.watch()