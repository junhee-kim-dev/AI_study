

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
transf = tr.Compose([tr.Resize(56), tr.ToTensor()])
# ToTensor => 토치 텐서로 바꾸기 + MinMaxScaler

from torchvision.datasets import MNIST
path = './Study25/_data/torch/'
train_dataset = MNIST(path, train=True, download=True, transform=transf)
test_dataset = MNIST(path, train=False, download=True, transform=transf)

# img_tensor, label = train_dataset[0]
# print(label)    # 5
# print(img_tensor.shape) # torch.Size([1, 56, 56])
# print(img_tensor.min(), img_tensor.max())   # tensor(0.) tensor(0.9922)

# exit()
# # #1. 데이터

# x_train, y_train = train_dataset.data / 255. , train_dataset.targets
# x_test, y_test = test_dataset.data / 255. , test_dataset.targets

# x_train = x_train.view(-1, x_train.shape[1]*x_train.shape[2])
# x_test = x_test.view(-1, x_test.shape[1]*x_test.shape[2])

# x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
# x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
# y_train = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
# y_test = torch.tensor(y_test, dtype=torch.long).to(DEVICE)

# train_set = TensorDataset(x_train, y_train)
# test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=32)

class CNN(nn.Module) :
    def __init__(self, num_features):
        # super().__init__()
        super(CNN, self).__init__()
        self.hl1 = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Dropout(0.3)
        )
        self.hl2 = nn.Sequential(
            nn.LazyConv2d(32, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Dropout(0.2)
        )
        self.hl3 = nn.Sequential(
            nn.LazyConv2d(16, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Dropout(0.2)
        )
        self.fla = nn.Flatten()
        self.hl4 = nn.Sequential(
            nn.LazyLinear(64),
            nn.ReLU()
        )
        self.hl5 = nn.Sequential(
            nn.LazyLinear(32),
            nn.ReLU()
        )
        self.out = nn.LazyLinear(10)

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
trainer.fit(10)
loss, acc = trainer.evaluate()
print(f"[Test] Loss: {loss:.4f} | Acc: {acc:.4f}")





