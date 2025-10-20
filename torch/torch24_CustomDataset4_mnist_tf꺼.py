import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from keras.datasets import mnist
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

x_train, y_train, x_test, y_test = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28).astype(np.float32) / 255.0
x_test = x_test.reshape(-1, 28, 28).astype(np.float32) / 255.0 

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



