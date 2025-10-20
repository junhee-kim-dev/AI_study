import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

# class MyDataset(Dataset):
#     def __init__(self):
#         self.x = [[1.0],[2.0],[3.0],[4.0],[5.0]]
#         self.y = [0,1,0,1,0]

#     def __len__(self):
#         return len(self.x)

#     def __getitem__(self, idx):
#         return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])
x = [[1.0],[2.0],[3.0],[4.0],[5.0]]
y = [0,1,0,1,0]

dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.int64))

loader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch_idx, (x, y) in enumerate(loader):
    print(f"========Batch {batch_idx}:=========")
    print(f"x: {x}")
    print(f"y: {y}")
    print()
    
# ========Batch 0:=========
# x: tensor([[1.],
#         [2.]])
# y: tensor([0, 1])

# ========Batch 1:=========
# x: tensor([[4.],
#         [3.]])
# y: tensor([1, 0])

# ========Batch 2:=========
# x: tensor([[5.]])
# y: tensor([0])
##########################################
# ========Batch 0:=========
# x: tensor([[4.],
#         [5.]])
# y: tensor([1, 0])

# ========Batch 1:=========
# x: tensor([[1.],
#         [3.]])
# y: tensor([0, 0])

# ========Batch 2:=========
# x: tensor([[2.]])
# y: tensor([1])