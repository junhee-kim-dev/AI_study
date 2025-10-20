import numpy as np
import matplotlib.pyplot as plt

def mish(x) :
    return x * np.tanh(np.log(1+np.exp(x)))

# relu = lambda x : x * np.tanh(np.log(1+np.exp(x)))
x = np.arange(-5, 5, 0.1)
y = mish(x)

plt.plot(x, y)
plt.grid()
plt.show()
