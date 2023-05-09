import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
from keras.models import load_model
from keras.datasets import mnist 
import cv2


model = load_model('model.keras')
model.summary()

ii = cv2.imread("test.png", 0)
img = cv2.resize(ii, (28, 28))
img = np.expand_dims(img, axis=-1)
im2arr = np.array(img)
im2arr = im2arr.reshape(1, 28, 28, 1)


prediction = model.predict(im2arr)
predictions = np.argmax(prediction)
print(predictions)
plt.imshow(img.squeeze(), cmap='gray')
plt.title(predictions)
plt.show()