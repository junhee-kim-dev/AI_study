
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

x = np.array([range(10)]).T
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [10,9,8,7,6,5,4,3,2,1],
              [9,8,7,6,5,4,3,2,1,0]]).T
x_pred = np.array([[10]])
# print(x.shape)  #(10, 1)
# print(y.shape)  #(10, 3)
# print(x_pred.shape) (1, 1)
# exit()
x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
y = torch.tensor(y, dtype=torch.float32).to(DEVICE)
x_pred = torch.tensor(x_pred, dtype=torch.float32).to(DEVICE)

x_scaled = (x - torch.mean(x)) / torch.std(x)
x_pred_scaled = (x_pred - torch.mean(x)) / torch.std(x)

model = nn.Sequential(
    nn.Linear(1,5),
    nn.Linear(5,4),
    nn.Linear(4,3),
    nn.Linear(3,3),
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y) :
    model.train()
    optimizer.zero_grad()
    hypo = model(x)
    loss = criterion(y, hypo)
    loss.backward()
    optimizer.step()
    return loss.item()

epochs=1000
for epoch in range(1, epochs+1) :
    loss = train(model, criterion, optimizer, x_scaled, y)
    print(f"epoch : {epochs}")
    print(f" loss : {loss:.6f}\n")
    
def evaluate(model, criterion, x, y) :
    model.eval()
    with torch.no_grad() :
        y_pred = model(x)
        f_loss = criterion(y, y_pred)
    return f_loss.item()

f_loss = evaluate(model, criterion, x_scaled, y)
results = model(x_pred_scaled)

print(np.round(f_loss, 5))
print(np.round(results.detach().cpu().numpy(),5))
# 0.0
# [[11. -0. -1.]]
