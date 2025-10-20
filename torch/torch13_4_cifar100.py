from torchvision.datasets import CIFAR100
path = './Study25/_data/torch/'

train_dataset = CIFAR100(path, train=True, download=True)
test_dataset = CIFAR100(path, train=False, download=True)

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


x_train, y_train = train_dataset.data / 255. , train_dataset.targets
x_test, y_test = test_dataset.data / 255. , test_dataset.targets

x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.long).to(DEVICE)

x_train = x_train.reshape(-1, 3072)
x_test = x_test.reshape(-1, 3072)

train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

class DNN(nn.Module) :
    def __init__(self, num_features):
        super().__init__()
        self.hl1 = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.hl2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.hl3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.hl4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.out = nn.Linear(128, 100)
        
    def forward(self, x) :
        x = self.hl1(x)
        x = self.hl2(x)
        x = self.hl3(x)
        x = self.hl4(x)
        x = self.out(x)
        return x
    
class Trainer :
    def __init__(self, model, criterion, optimizer, train_loader, test_loader):
        self.model = model.to(DEVICE)
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        
    def train(self) :
        self.model.train()
        epoch_loss = 0
        epoch_acc = 0
        for x, y in self.train_loader :
            self.optimizer.zero_grad()
            hypo = self.model(x)
            loss = self.criterion(hypo, y)
            loss.backward()
            self.optimizer.step()
            epoch_loss +=loss.item()
            y_pred = torch.argmax(hypo, 1)
            acc = (y_pred ==y).float().mean()
            epoch_acc += acc.item()
            
        return epoch_loss / len(self.train_loader), epoch_acc / len(self.train_loader)
    
    def evaluate(self) :
        self.model.eval()
        epoch_loss = 0
        epoch_acc = 0
        with torch.no_grad() :
            for x, y in self.test_loader :
                hypo = self.model(x)
                loss = self.criterion(hypo, y)
                
                y_pred = torch.argmax(hypo, 1)
                acc = (y_pred == y).float().mean()
                epoch_loss += loss.item()
                epoch_acc += acc.item()
        return epoch_loss / len(self.test_loader), epoch_acc / len(self.test_loader)

    def fit(self, epochs=100) :
        for epoch in range(epochs +1) :
            loss, acc = self.train()
            val_loss, val_acc = self.evaluate()
            print(f"[Epoch {epoch}] Loss: {loss:.4f} | Acc: {acc:.4f} | Val_Loss: {val_loss:.4f} | Val_Acc: {val_acc:.4f}")

model = DNN(3072)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
trainer = Trainer( model, criterion, optimizer, train_loader, test_loader)
trainer.fit()
loss, acc = trainer.evaluate()
print(f"[Test] Loss: {loss:.4f} | Acc: {acc:.4f}")

# [Test] Loss: 4.0348 | Acc: 0.0726