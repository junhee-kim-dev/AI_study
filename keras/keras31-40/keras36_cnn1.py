# 커널 사이즈로 점점 축소시킴
# 너무 많이 자르면 너무 소실되면서 loss가 커짐
# 너무 조금 자르면 특성값이 크지 않아서 loss가 커짐
# 그러니까 적당히 자르는 것이 하이퍼 파라미터 튜닝임

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D

# 원본이 (N,5,5,1) 이미지라면
# 5,5 짜리 흑백 사진이 N장 있는것
# 원본이 (N,5,5,3) 이미지라면
# 5,5 짜리 컬러 사진이 N장
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(5,5,1))) # 10: 노드 수 -> 이만큼 그림을 증폭
                            # (2,2): 커널 사이즈 /결과적으로 (N, 4,4,10)을 출력
model.add(Conv2D(5, (2,2))) # (4,4,10)을 받아서 (3,3,5)를 출력
model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 4, 4, 10)          50        
#  conv2d_1 (Conv2D)           (None, 3, 3, 5)           205       
# =================================================================
# Total params: 255
# Trainable params: 255
# Non-trainable params: 0
# _________________________________________________________________