import numpy as np

data = np.random.randn(50)
n, bins = np.histogram(data, len(data))

print(data)
print(n)
print(bins)