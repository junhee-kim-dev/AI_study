
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])

x_test  = np.array([8,9,10,11])
y_test  = np.array([8,9,10,11])

x_pre = np.array([12,13,14])

x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)

x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)

x_pre = torch.tensor(x_pre, dtype=torch.float32).unsqueeze(1).to(DEVICE)

x_train_scaled = (x_train - torch.mean(x_train)) / torch.std(x_train)
x_test_scaled = (x_test - torch.mean(x_train)) / torch.std(x_train)
x_pre_scaled = (x_pre - torch.mean(x_train)) / torch.std(x_train)

model = nn.Sequential(
    nn.Linear(1,5),
    nn.Linear(5,4),
    nn.Linear(4,3),
    nn.Linear(3,2),
    nn.Linear(2,1),
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

for i in range(1000) :
    loss = train(model, criterion, optimizer, x_train_scaled, y_train)
    print(f"{i+1} epoch -> {loss:.6f}")

def evaluate(model, criterion, x, y) :
    model.eval()
    with torch.no_grad() :
        y_pred = model(x)
        f_loss = criterion(y, y_pred)
    return f_loss.item()

f_loss = evaluate(model, criterion, x_test_scaled, y_test)
results = model(x_pre_scaled)
results_ = np.round(results.detach().cpu().numpy(), 4)
results__ = results_.reshape(-1,)

print(np.round(f_loss, 4))
print(results__)

# 0.0
# [12. 13. 14.]