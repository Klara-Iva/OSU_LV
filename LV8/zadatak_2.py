import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from keras.datasets import mnist 


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)


model = load_model('model.keras')
model.summary()

(x_train, y_train), (x_test,y_test) = keras.datasets.mnist.load_data()


# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

# pretvori labele
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)

predictions = model.predict(x_test_s)

y_test_s = np.argmax(y_test_s, axis = 1)
predictions = np.argmax(predictions, axis = 1)

wrong_predicted = x_test_s[y_test_s != predictions]
print("Wrong predictions:" , len(wrong_predicted))

wrong_y_test_s = y_test_s[y_test_s != predictions]
wrong_predictions = predictions[y_test_s != predictions]

for i in range(10,20,2):
    plt.figure()
    plt.title("Stvarna vrijednost: " + str(wrong_y_test_s[i]) +  " Predpostavljena vijednost:" + str(wrong_predictions[i]))
    plt.imshow(wrong_predicted[i])  
    plt.show()