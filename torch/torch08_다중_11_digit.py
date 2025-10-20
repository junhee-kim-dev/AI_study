

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
import time, datetime
import matplotlib.pyplot as plt

# 시드 고정
seed = 123
import random
random.seed(seed)
np.random.seed(seed)
# tf.random.set_seed(seed)

datasets = load_digits()
x = datasets.data
y = datasets.target

print(x.shape)
print(np.unique(y, return_counts=True))
# exit()

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

USE_DEVICE = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_DEVICE else 'cpu')

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed, stratify=y
)

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.long).to(DEVICE)

model = nn.Sequential(
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.Linear(16, 10),
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, x, y) :
    model.train()
    optimizer.zero_grad()
    hypo = model(x)
    loss = criterion(hypo, y)
    loss.backward()
    optimizer.step()
    return loss.item()

for i in range(1000) :
    loss = train(model, criterion, optimizer, x_train, y_train)
    print(f"epoch : {i+1} / loss : {loss} ")

def evaluate(model, criterion, x, y) :
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        f_loss = criterion(y_pred, y)
    return f_loss.item()

f_loss = evaluate(model, criterion, x_test, y_test)
# y_pred = model(x_test)
y_pred = model(x_test)
y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()
y_test = y_test.cpu().numpy()
acc = accuracy_score(y_test, y_pred)

print(acc)
# 0.9888888888888889
