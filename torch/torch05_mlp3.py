
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

print("torch :", torch.__version__, "사용 DEVICE :", DEVICE)
# torch : 2.7.1+cu128 사용 DEVICE : cuda

#1. 데이터
x = np.array([range(10), range(21,31), range(201,211)]).transpose()
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [10,9,8,7,6,5,4,3,2,1]]).T

# x = torch.FloatTensor(x).to(DEVICE)
# y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)

x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
y = torch.tensor(y, dtype=torch.float32).to(DEVICE)

x_scaled = (x - torch.mean(x)) / torch.std(x)

model = nn.Sequential(
    nn.Linear(3, 5),
    nn.Linear(5, 4),
    nn.Linear(4, 3),
    nn.Linear(3, 2),
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, x, y):
    model.train()
    optimizer.zero_grad()
    hypo = model(x)
    loss = criterion(hypo, y)
    loss.backward()
    optimizer.step()
    
    return loss.item()

epochs=1000
for epoch in range(1, epochs+1) :
    loss = train(model, criterion, optimizer, x_scaled, y)
    print(f"epoch : {epochs}")
    print(f" loss : {loss:.6f}\n")

def eval(model, criterion, x, y) :
    model.eval()
    
    with torch.no_grad() :
        y_pred = model(x)
        final_loss = criterion(y, y_pred)
    
    return final_loss.item()
        
final_loss = eval(model, criterion, x_scaled, y)
print(f"최종 손실 : {final_loss:.6f}")

x_pred = torch.tensor([[10, 1.9, 0]]).to(DEVICE)
x_pred_scaled = (x_pred - torch.mean(x)) / torch.std(x)

f_loss = eval(model, criterion, x_pred_scaled, y)
result = model(x_pred_scaled)
print(f_loss)   #13.772470474243164
# print(result.item())    #RuntimeError: a Tensor with 2 elements cannot be converted to Scalar
# print(result.detach())    #tensor([[3.4671, 3.2514]], device='cuda:0')
# print(result.detach().cpu())    #tensor([[2.5072, 3.0193]])
print(result.detach().cpu().numpy())    #[[2.8716996 2.9638286]]
