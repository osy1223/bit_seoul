from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# 현재 시각에서 3시간 이전의 시각 구하기
edt = datetime.now() - timedelta(hours=3)
print(edt)
# 2020-12-01 16:27:47.220016

# 현재 시각에서 3초전 이전의 시각 구하기
edt = datetime.now() - timedelta(seconds=3)
print(edt)
# 2020-12-01 19:28:52.263895
########## months는 없습니다 ##########

edt = datetime.now() - relativedelta(months=3)
print(edt)
# 2020-09-01 19:30:00.121291
########## 그럴땐 relativedelta 이용 ##########