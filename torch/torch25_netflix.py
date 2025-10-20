import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#1. 데이터
data_path = './Study25/_data/kaggle/netflix/'
train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
submission = pd.read_csv(data_path +'sample_submission.csv')

from torch.utils.data import TensorDataset, DataLoader, Dataset

class Custom_Dataset(Dataset):
    def __init__(self, df, timesteps):
        self.train = df
        self.timesteps = timesteps
        self.x = self.train.iloc[:, 1:4].values.astype(np.float32)
        self.x = (self.x - np.min(self.x, axis=0)) / (np.max(self.x, axis=0) - np.min(self.x, axis=0))
        self.y = self.train['Close'].values.astype(np.float32)

    # (967, 3) -> (n, 30, 3)시계열로 변경
    def __len__(self) :
        return len(self.x) - self.timesteps #행 - timesteps
    
    def __getitem__(self, index):
        x = self.x[index : index+self.timesteps]    # x[idx : idx + timesteps]
        y = self.y[index+self.timesteps]            # y[idx + timesteps]
        
        return x, y

custom_dataset = Custom_Dataset(train, 30)
# for x, y in iter(custom_dataset) :
#     print(type(x), type(y))   #<class 'numpy.ndarray'> <class 'numpy.int64'>
#     break

# exit()
# print(type(x))

train_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True)

#2. 모델
class RNN(nn.Module) :
    def __init__(self):
        super().__init__()
        
        self.rnn_layer = nn.RNN(
            input_size=3, 
            hidden_size=64, 
            num_layers=3,
            batch_first=True
            )
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x) :
        x, _ = self.rnn_layer(x)
        # x = torch.reshape(x, (-1, 30*64))
        x = x[:,-1,:]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
model = RNN().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


for epoch in range(1,201) :
    iterator = tqdm(train_loader)
    total_loss = 0
    for x, y in iterator :
        x,y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        
        hypothesis = model(x)
        loss = criterion(hypothesis, y)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
        iterator.set_description(f'[Epoch {epoch}] Loss {total_loss / len(train_loader):.4f}')
    # print(f"[Epoch {epoch}] Loss {loss}")

save_path = './Study25/_save/torch/'
torch.save(model.state_dict(), save_path + 'netflix.pth')


