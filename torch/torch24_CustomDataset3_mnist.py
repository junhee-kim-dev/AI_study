import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np
import random

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

x_train, y_train = train_dataset.data / 255., train_dataset.targets
x_test, y_test = test_dataset.data / 255., test_dataset.targets

class MyDataset(Dataset):
    def __init__(self, x_train, y_train):
        super().__init__()
        self.x = x_train
        self.y = y_train
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx].unsqueeze(0), self.y[idx]

train_dataset = MyDataset(x_train, y_train)
test_dataset = MyDataset(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)



