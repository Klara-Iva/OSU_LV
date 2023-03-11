import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("road.jpg")
img = img[:, :, 0].copy()


print(img.shape)
print(img.dtype)

plt.figure()

plt.title("brighter")
plt.imshow(img, vmin=10, vmax=100, cmap="gray")
plt.show()

plt.title("cropped")
plt.imshow(img[0:, 160:320], cmap="gray")
plt.show()

plt.title("rotated")
plt.imshow(np.rot90(img, 3), cmap="gray")
plt.show()

plt.imshow(np.fliplr(img), cmap="gray")
plt.title("mirrored")
plt.show()
