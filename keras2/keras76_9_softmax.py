
import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

# softmax = lambda x, alpha : (x>0)*x + (x<0)*(alpha*(np.exp(x)-1))

x = np.arange(1, 5, 1)

y = softmax(x)

ratio=y
labels=y
plt.pie(ratio, labels, shadow=True, startangle=90)

plt.grid()
plt.show()