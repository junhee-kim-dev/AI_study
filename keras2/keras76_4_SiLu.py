import numpy as np
import matplotlib.pyplot as plt

def silu(x) :
    return x * (1 / (1+np.exp(-x)))

# relu = lambda x : np.maximum(0, x)
x = np.arange(-5, 5, 0.1)
y = silu(x)

plt.plot(x, y)
plt.grid()
plt.show()
