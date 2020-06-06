import numpy as np

import matplotlib.pyplot as plt

x = np.arange(0, 5, 1)
y = np.exp(x)

y_sum = np.sum(y)
y = y/y_sum

plt.plot(x, y)
plt.show()
