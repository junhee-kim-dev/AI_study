from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
import numpy as np

a = np.array(range(1,101))
x_pred = np.array(range(96,106))

timesteps=11

def split_1d(dataset, timesteps):
    all = []
    for i in range(len(dataset) - timesteps +1) :
        subset = dataset[i : (i+timesteps)]
        all.append(subset)
    all = np.array(all)
    x = all[:,:-1]
    y = all[:,-1]
    return x, y

x, y = split_1d(a, timesteps=timesteps)

print(x.shape, y.shape) #(90, 10) (90,)
print(x, y)


10-4 = 6
1,2,3,4,5,6,7,8,9,10,11,12,13,14

10 - 4 + 1
