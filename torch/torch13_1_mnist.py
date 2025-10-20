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

#1. 데이터
from torchvision.datasets import MNIST
path = './Study25/_data/torch/'

train_dataset = MNIST(path, train=True, download=True)
test_dataset = MNIST(path, train=False, download=True)

# print(train_dataset)
# print(type(train_dataset))      # <class 'torchvision.datasets.mnist.MNIST'>
# print(train_dataset[0])         # (<PIL.Image.Image image mode=L size=28x28 at 0x7E6361A63F10>, 5)

x_train, y_train = train_dataset.data / 255., train_dataset.targets
x_test, y_test = test_dataset.data / 255., test_dataset.targets
# print(x_train)
# print(y_train)
# print(x_train.size(), y_train.size())   # torch.Size([60000, 28, 28]) torch.Size([60000])
# print(torch.min(x_train))   # tensor(0, dtype=torch.uint8)
# print(torch.max(x_train))   # tensor(255, dtype=torch.uint8)

x_train, x_test = x_train.view(-1, 28*28), x_test.view(-1, 784)
# print(x_train.shape , x_test.size())    torch.Size([60000, 784]) torch.Size([10000, 784])

x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_set, shuffle=True, batch_size=100)
test_loader = DataLoader(test_set, shuffle=False, batch_size=100)

class DNN(nn.Module) :
    def __init__(self, num_features):
        super().__init__()
        # super(DNN, self).__init__()
        
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU()
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.hidden_layer5 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(32, 10)
        
    def forward(self, x) :
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer4(x)
        x = self.hidden_layer5(x)
        x = self.output_layer(x)
        return x

model = DNN(784).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.1e-4)

def fit(model, criterion, optimizer, loader) :
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for x, y in loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        
        hypo = model(x)
        loss = criterion(hypo, y)
        
        loss.backward()
        optimizer.step()    # w = w - lr * 기울기
        
        epoch_loss += loss.item()
        
        y_pred = torch.argmax(hypo, 1)
        acc = (y_pred == y).float().mean()
        
        epoch_acc += acc.item()

    total_acc = epoch_acc / len(loader)
    total_loss = epoch_loss / len(loader)
    return total_loss, total_acc

def evaluate(model, criterion, loader) :
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad() :
        for x, y in loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            
            hypo = model(x)
            loss = criterion(hypo, y)
            
            
            y_pred = torch.argmax(hypo, 1)
            acc = (y_pred == y).float().mean()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()

        total_acc = epoch_acc / len(loader)
        total_loss = epoch_loss / len(loader)

    return total_loss, total_acc

EPOCHS = 20
for i in range(EPOCHS) :
    loss, acc = fit(model, criterion, optimizer, train_loader)
    val_loss, val_acc = evaluate(model, criterion, test_loader)
    print(f"[EPOCH {i+1}] loss {loss:.4f} | acc {acc:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")

f_loss, f_acc = evaluate(model, criterion, test_loader)
print("="*15)
print(f"[Test] loss {f_loss:.4f} | acc {f_acc:.4f}")

# [Test] loss 0.0001 | acc 1.0000










