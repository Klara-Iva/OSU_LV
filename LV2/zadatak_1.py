import numpy as np
import matplotlib.pyplot as plt

y = np.array([1.0, 2.0, 2.0, 1.0, 1.0])
x = np.array([1.0, 2.0, 3.0, 3.0, 1.0])

plt.plot(x, y, 'y', linewidth=4, marker="X", markersize=20)
plt.axis([0.0, 4.0, 0.0, 4.0])
plt.xlabel('x os')
plt.ylabel('y os')
plt.title('primjer')
plt.show()
