
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

x = np.array(range(100)).reshape(-1,1)
y = np.array(range(1,101)).reshape(-1,1)
x_pred = np.array([101, 102]).reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=42, train_size=0.8
)

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
x_pred = ss.transform(x_pred)

x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
x_pred = torch.tensor(x_pred, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

model = nn.Sequential(
    nn.Linear(1, 30),
    nn.Linear(30, 20),
    nn.Linear(20, 10),
    nn.Linear(10,1),
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

for j in range(2000):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print(f"{j+1} epochs loss = {loss:.6f}")
    
def evaluate(model, criterion, x, y) :
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        f_loss = criterion(y, y_pred)
    return f_loss.item()

f_loss = evaluate(model, criterion, x_test, y_test)
f_loss = np.round(f_loss, 6)
result = model(x_pred)
result = np.round(result.detach().cpu().numpy(), 4).reshape(-1,)
print(f_loss)
print(result)
# 0.0
# [102. 103.]