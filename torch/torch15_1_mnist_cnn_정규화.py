

import warnings
warnings. filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from torch.utils.data import DataLoader, TensorDataset

USE_DEVICE = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_DEVICE else 'cpu')

SEED = 1

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

import torchvision.transforms as tr
transf = tr.Compose([tr.Resize(56), tr.ToTensor(), tr.Normalize((0.5,), (0.5,))])
# ToTensor => 토치 텐서로 바꾸기 + MinMaxScaler
# Z-Score Normalization = 이거 정규화는 평균이 0, 표준편차가 1이 되도록 변환
# (x - mean) / std
# x - 0.5 / 0.5  통상 평균 0.5, 표준편차 0.5로 정규화 -> -1 ~ 1 사이로 변환

from torchvision.datasets import MNIST
path = './Study25/_data/torch/'
train_dataset = MNIST(path, train=True, download=True, transform=transf)
test_dataset = MNIST(path, train=False, download=True, transform=transf)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=32)

class CNN(nn.Module) :
    def __init__(self, num_features):
        # super().__init__()
        super(CNN, self).__init__()
        self.hl1 = nn.Sequential(
            nn.Conv2d(num_features, 64, kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Dropout(0.3)
        )
        self.hl2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Dropout(0.2)
        )
        self.hl3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Dropout(0.2)
        )
        self.hl4 = nn.Sequential(
            nn.Linear(16*5*5, 64),
            nn.ReLU()
        )
        self.hl5 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.out = nn.Linear(32, 10)
        self.fla = nn.Flatten()

    def forward(self, x) :
        x = self.hl1(x)
        x = self.hl2(x)
        x = self.hl3(x)
        x = self.fla(x)
        x = self.hl4(x)
        x = self.hl5(x)
        x = self.out(x)
        return x

class Trainer:
    def __init__(self, model, criterion, optimizer, train_loader, val_loader):
        self.model = model.to(DEVICE)
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train(self):
        self.model.train()
        epoch_loss = 0
        epoch_acc = 0
        for x, y in self.train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            self.optimizer.zero_grad()
            hypo = self.model(x)
            loss = self.criterion(hypo, y)
            loss.backward()
            self.optimizer.step()

            y_pred = torch.argmax(hypo, dim=1)
            acc = (y_pred == y).float().mean()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(self.train_loader), epoch_acc / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        epoch_loss = 0
        epoch_acc = 0
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                hypo = self.model(x)
                loss = self.criterion(hypo, y)

                y_pred = torch.argmax(hypo, dim=1)
                acc = (y_pred == y).float().mean()

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(self.val_loader), epoch_acc / len(self.val_loader)

    def fit(self, epochs):
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train()
            val_loss, val_acc = self.evaluate()
            print(f"[Epoch {epoch}] Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Val_Loss: {val_loss:.4f} | Val_Acc: {val_acc:.4f}")

model = CNN(1)  # nn.Module 에서 conv2d는 channel만 input으로 넣어줌
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.1e-4)

trainer =Trainer(model, criterion, optimizer, train_loader, test_loader)
trainer.fit(100)

loss, acc = trainer.evaluate()
print(f"[Test] Loss: {loss:.4f} | Acc: {acc:.4f}")





