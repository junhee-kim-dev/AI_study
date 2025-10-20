import numpy as np

data = [1,2,3,4,5]
print(np.percentile(data, 25))

data = [10,20,30,40]
print(np.percentile(data, 25))

'''
index의 위치 찾기
rank = (n - 1) * (q / 100)

보간법
작은값 = data의 0번째 = 10
  큰값 = data의 1번쨰 = 20

백분위값 = 작은값 + (큰값 - 작은값) * rank
         = 10 + (20 - 10) * 0.75
         = 10 + 7.5 = 17.5
'''

