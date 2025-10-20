import numpy as np
import matplotlib.pyplot as plt

data = np.random.exponential(scale=1000.0, size=1000)
# print(data)
# print(data.shape)                   # (1000,)
print(np.min(data), np.max(data))   # 1.0609065099947392 8604.99604730902

log_data = np.log1p(data)       # log1p : log0 이 되는 불상사를 막기 위해 +1 해줌 / 원상복귀는 np.expm1p(data)
# print(log_data)
print(np.min(log_data), np.max(log_data))   # 0.7231459394018943 9.060214454103647

plt.subplot(1,2,1)
plt.hist(data, bins=50, color='blue', alpha=0.5)
plt.title('Original')

plt.subplot(1,2,2)
plt.hist(log_data, bins=50, color='red', alpha=0.5)
plt.title('Log Transformed')

plt.show()

# re_data = np.expm1(log_data)

# print(re_data)