import torch
from torch.utils.data import Dataset, DataLoader 

class MyDataset(Dataset):
    def __init__(self):
        self.x = [[1.0],[2.0],[3.0],[4.0],[5.0]]
        self.y = [0,1,0,1,0]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])
    
dataset = MyDataset()    

loader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch_idx, (x, y) in enumerate(loader):
    print(f"========Batch {batch_idx}:=========")
    print(f"x: {x}")
    print(f"y: {y}")
    print()
    
#     ========Batch 0:=========
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