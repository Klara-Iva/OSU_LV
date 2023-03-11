import numpy as np
import matplotlib.pyplot as plt

black = np.ones((50, 50))
white = np.zeros((50, 50))

column2 = np.vstack((black, white))
column1 = np.vstack((white, black))
img = np.hstack((column1, column2))
plt.imshow(img, cmap="gray")
plt.show()
