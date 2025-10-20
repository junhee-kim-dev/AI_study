import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as tr
from torchvision.datasets import MNIST
import random
import numpy as np
import pandas as pd
USE_DEVICE = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_DEVICE else 'cpu')

datasets = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9]])
y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape)
x = x.reshape(x.shape[0], x.shape[1], 1)

print(x.shape) # (7, 3, 1)

x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
y = torch.tensor(y, dtype=torch.float32).to(DEVICE)

print(x.shape, y.shape)  # (7, 3, 1) torch.Size([7])

from torch.utils.data import TensorDataset
train_set = TensorDataset(x, y)
train_loader = DataLoader(train_set, batch_size=2, shuffle=True)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.cell = nn.RNN(
            input_size=1,   # 피쳐의 개수
            hidden_size=32, # 아웃풋 노드의 개수
            num_layers=1,   # default=1, RNN 층의 개수
            batch_first=True# default=False, True면 (batch, time, feature) 순서로 입력 반드시 쓴다
        )
        self.fc1 = nn.Linear(3*32, 16)
        self.fc2 = nn.Linear(16,8)
        self.fc3 = nn.Linear(8,1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out, _ = self.cell(x)
        out = self.relu(out)
        out = out.reshape(-1, 32 * 3)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


# from torchsummary import summary
# summary(model, input_size=(3, 1))
# from torchinfo import summary
# summary(model, input_size=(2, 3, 1))  # batch_size=2, time_step=3, feature=1
class Trainer:
    def __init__(self, model, criterion, optimizer, train_loader):
        self.model = model.to(DEVICE)
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
    
    def train(self, epochs):
        for epoch in range(epochs):
            total_loss = 0
            for x_batch, y_batch in self.train_loader:
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                self.optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = self.criterion(outputs.squeeze(), y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(self.train_loader):.4f}')
                
model = RNN().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00099)

trainer = Trainer(model, criterion, optimizer, train_loader)
trainer.train(epochs=100)

# Epoch [100/100], Loss: 0.0062