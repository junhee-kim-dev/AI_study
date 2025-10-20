import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

# 넘파이를 그대로 쓰지 않고 토치텐서로 씀
# x = torch.FloatTensor(x)
# print(x)         # tensor([1., 2., 3.])
# print(x.shape)   # torch.Size([3])
# print(x.size())  # torch.Size([3])

x = torch.FloatTensor(x).unsqueeze(1)         # .unsqueeze(1) -> (3,) : (3,1)
# x = torch.FloatTensor(x).unsqueeze(0)       # .unsqueeze(0) -> (3,) : (1,3)
# 텐서플로우에서는 벡터가 먹히지 않아. 항상 최소 매트릭스
# print(x)
# tensor([[1.],
#         [2.],
#         [3.]])
# print(x.shape)      # torch.Size([3, 1])
# print(x.size())     # torch.Size([3, 1])

y = torch.FloatTensor(y).unsqueeze(1)
# y도 당연히 unsqueeze
# print(y.size())     # torch.Size([3, 1])

#2. 모델구성
model = nn.Linear(1,1)
# nn.Linear(input, output)
# 이게 무슨 말이냐 하면
# model = Sequential()
# model.add(Dense(1, input_dim=1))
# 이라는 말임 

# y = wx+b 는 사실! 틀렸다! 사실! 거짓말이었다!
# 사실 y = xw+b임 
# 왜냐하면 x 데이터의 열 형태에 맞춰 가중치가 생성되어야 하기 때문
# if x = (N, 10) -> w = (10, N) 
# if x = (N, 8)  -> w = (8, N)

#3. 컴파일 훈련
# 기존: model.compile(loss='mse', optimizer='adam')
# torch 에서는 loss와 optimizer를 각각 정의해야 줘야함
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
# optimizer = optim.SGD(model.parameters(), lr=0.001)
# GD = Gradient Descent

def train(model, criterion, optimizer, x, y):
    model.train()         # default
    # [훈련모드] 드랍아웃, 배치노말 적용
    
    # 순전파
    optimizer.zero_grad()   # 기울기 초기화
                # 각 배치마다 기울기를 초기화(0으로)하여, 기울기 누적에 의한 문제 해결
                # 이걸 안 하면 계속 기울기가 커짐
    hypothesis = model(x)   # y = xw+b  hypothesis = 모델의 결과(y_predict)
    loss = criterion(hypothesis, y) # loss = mse = Sigma(y - hypothesis)^2 /n
    
    # 역전파
    loss.backward()         # 기울기(gradient)값까지만 계산.
    optimizer.step()        # 가중치 갱신

    # 이렇게 하면 1epoch 이자 1batch
    return loss.item()      # torch를 
    
epochs = 5000
for epoch in range(1, epochs+1) :
    loss = train(model=model, criterion=criterion, optimizer=optimizer, x=x, y=y)
    print(f'epoch: {epoch}, loss: {loss}')

# !!! 기울기와 가중치는 다르다 !!!
# 에러, 로스, 코스트, 에러 등등 다 같은 말임

print('=====================================')
#4. 평가 예측
# loss = model.evaluate() 이거 torch로 만들기
def evaluate(model, criterion, x, y) :
    model.eval()
    # [평가모드] 드랍아웃, 배치노말 절대 안 됨
    
    with torch.no_grad() : # 기울기 갱신을 하지 않겠다.
        y_pred = model(x)
        final_loss = criterion(y, y_pred)   # loss의 최종값
    
    return final_loss.item()

final_loss = evaluate(model=model, criterion=criterion, x=x, y=y)
print(f"최종 loss : {final_loss}")

result = model(torch.Tensor([[4]]))
# print(f"4의 예측값 : {result}")           
# # 4의 예측값 : tensor([[3.7224]], grad_fn=<AddmmBackward0>)
print(f"4의 예측값 : {result.item()}")
# 4의 예측값 : 3.614001750946045  4의 예측값 : 3.9999990463256836  4의 예측값 : 4.0







