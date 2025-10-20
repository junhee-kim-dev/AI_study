
'''

결측치 처리
1. 삭제 - 행 또는 열
2. 임의의 값
    - 0    : fillna(0)
    - 평균 : mean(이상치)
    - 중위 : median
    - 앞값 : ffill
    - 뒷값 : bfill
    - 특정값 : fillna(n) -> 조건 보고 넣는게 낫다.
    - 기타등등...
3. 보간(interpolate) - 알려진 데이터 점 집합의 범위 내에 새 데이터 점을 추가하는 기법
4. 모델 : pseudo , .predict, (전혀 다른 모델 사용.)
5. 부스팅 계열 모델 : 통상 이상치, 결측치에 대해 영향을 덜 받는다.

'''

import pandas as pd
import numpy as np

date = [
    '16/7/2025','17/7/2025','18/7/2025',
    '19/7/2025','20/7/2025','21/7/2025'
]

date = pd.to_datetime(date)
print(date)
# DatetimeIndex(['2025-07-16', '2025-07-17', '2025-07-18', '2025-07-19',
#                '2025-07-20', '2025-07-21'],
#               dtype='datetime64[ns]', freq=None)

print('===================================')
ts = pd.Series([2, np.nan, np.nan, 8, 10, np.nan], index=date)
print(ts)

print('===================================')

ts = ts.interpolate()
print(ts)
# 2025-07-16     2.0
# 2025-07-17     4.0
# 2025-07-18     6.0
# 2025-07-19     8.0
# 2025-07-20    10.0
# 2025-07-21    10.0

# 형변환을 자동으로 해준다.
# 중간값들을 linear로 채워준다. 마지막은 ffill




