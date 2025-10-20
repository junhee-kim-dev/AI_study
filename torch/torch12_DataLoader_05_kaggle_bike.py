import pandas as pd
path = './Study25/_data/kaggle/bike-sharing-demand/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)

x = train_csv.drop(['count'], axis=1)
y = train_csv[['count']]

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch import float32 as f32
from torch import long

import random
SEED = 337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

use = torch.cuda.is_available()
dev = torch.device('cuda' if use else 'cpu')

# print(x.shape, y.shape)
# exit()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=333, train_size=0.8
)

ss = StandardScaler()
ss.fit(x_train)
x_train = ss.transform(x_train)
x_test = ss.transform(x_test)

x_train = torch.tensor(x_train, dtype=f32).to(dev)
x_test = torch.tensor(x_test, dtype=f32).to(dev)
y_train = torch.tensor(y_train.values, dtype=f32).to(dev)
y_test = torch.tensor(y_test.values, dtype=f32).to(dev)

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_set, shuffle=True, batch_size=100)
test_loader = DataLoader(test_set, shuffle=True, batch_size=100)

class Model(nn.Module) :
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        self.l1 = nn.Linear(input_dim, 128)
        self.l2 = nn.Linear(128,64)
        self.l3 = nn.Linear(64,32)
        self.l4 = nn.Linear(32, output_dim)
        self.rl = nn.ReLU()

    def forward(self, x) :
        x = self.l1(x)
        x = self.rl(x)
        x = self.l2(x)
        x = self.rl(x)
        x = self.l3(x)
        x = self.rl(x)
        x = self.l4(x)
        return x

model = Model(10, 1).to(dev)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def fit(model, criterion, optimizer, loader) :
    model.train()
    total_loss = 0
    for x,y in loader :
        optimizer.zero_grad()
        hypo = model(x)
        loss = criterion(hypo, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    tloss = total_loss / len(loader)
    return tloss

for i in range(100) :
    verbose = fit(model, criterion, optimizer, train_loader)
    print(f"epoch : {i+1} / loss : {verbose:.8f}")

def evaluate(model, criterion, loader) :
    model.eval()
    total_loss =0
    with torch.no_grad() :
        for x, y in loader:
            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item()
    tloss = total_loss / len(loader)
    return tloss

loss = evaluate(model, criterion, test_loader)
result = model(x_test).cpu().detach().numpy()
y_test = y_test.cpu().detach().numpy()
r2 = r2_score(y_test, result)

print(f"\n최종 LOSS : {loss:.5f}")
print(f"최종   R2 : {r2:.5f}")

# 최종 LOSS : 11.02897
# 최종   R2 : 0.99965

# 최종 LOSS : 8.39117
# 최종   R2 : 0.99973

# 최종 LOSS : 0.92525
# 최종   R2 : 0.99997