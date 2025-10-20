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
        self.x = self.train.iloc[:, 1:4].values
        self.x = (self.x - np.min(self.x, axis=0)) / (np.max(self.x, axis=0) - np.min(self.x, axis=0))
        self.y = self.train['Close'].values

    # (967, 3) -> (n, 30, 3)시계열로 변경
    def __len__(self) :
        return len(self.x) - self.timesteps #행 - timesteps
    
    def __getitem__(self, index):
        x = self.x[index : index+self.timesteps]    # x[idx : idx + timesteps]
        y = self.y[index+self.timesteps]            # y[idx + timesteps]
        return torch.tensor(x, dtype=torch.float32).to(DEVICE),torch.tensor(y, dtype=torch.float32).to(DEVICE)

custom_dataset = Custom_Dataset(train, 30)
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


# class Trainer:
#     def __init__(self, model, criterion, optimizer, train_loader):
#         self.model = model.to(DEVICE)
#         self.criterion = criterion
#         self.optimizer = optimizer
#         self.train_loader = train_loader
    
#     def train(self):
#         self.model.train()
#         epoch_loss = 0

#         iterator = tqdm(self.train_loader)  # 진행바
#         for x, y in iterator:
#             self.optimizer.zero_grad()
#             hypo = self.model(x)
#             loss = self.criterion(hypo, y)
#             loss.backward()
#             self.optimizer.step()

#             epoch_loss += loss.item()

#         return epoch_loss / len(self.train_loader)
    
#     def evaluate(self) :
#         self.model.eval()
#         epoch_loss = 0
        
#         with torch.no_grad() :
#             for x, y in self.train_loader:
#                 hypo = self.model(x)
#                 loss = self.criterion(hypo, y)
#                 epoch_loss += loss.item()
        
#         return epoch_loss / len(self.train_loader)
    
#     def fit(self, epochs=10) :
#         for epoch in range(epochs) :
#             loss = self.train()
#             print(f"[Epoch {epoch+1} | Loss {loss:.6f}]")

model = RNN().to(DEVICE)
# trainer = Trainer(model, nn.MSELoss(), optim.Adam(RNN().parameters(), lr=0.0007), train_loader)
# trainer.fit(20)

save_path = './Study25/_save/torch/'
import os
os.makedirs(save_path, exist_ok=True)
# torch.save(model.state_dict(), save_path +'t25.pth')

y_predict = []
y_true = []
total_loss = 0

with torch.no_grad() :
    model.load_state_dict(torch.load(save_path + 'netflix.pth', map_location=DEVICE))
    
    for x_test, y_test in train_loader :
        y_test = y_test.unsqueeze(1)
        y_pred = model(x_test)
        y_predict.append(y_pred.cpu().numpy())
        y_true.append(y_test.cpu().numpy())
        
        loss = nn.MSELoss()(y_pred, y_test)
        total_loss += loss.item() / len(train_loader)
        
    print(f"Val_Loss {total_loss}")

# print(y_predict)
y_predict = np.concatenate(y_predict).flatten()
# print(y_predict.shape)
y_true = np.concatenate(y_true).flatten()

from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_predict)
print(r2)
