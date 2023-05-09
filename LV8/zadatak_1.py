import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# prikaz karakteristika train i test podataka
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# prikazi nekoliko slika iz train skupa
for i in range(11,19,1) :
    plt.subplot(330+1+i-11)
    plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
plt.show()

# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")


# pretvori labele
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)

# kreiraj model pomocu keras.Sequential(); prikazi njegovu strukturu
model = keras.Sequential()
model.add(layers.Flatten(input_shape=input_shape))
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(50,activation="sigmoid"))
model.add(layers.Dense(10, activation="softmax"))
model.summary()

#definiraj karakteristike procesa ucenja pomocu .compile()
model.compile ( loss ="categorical_crossentropy" ,
    optimizer ="adam",
    metrics = ["accuracy" ,])


# provedi ucenje mreze
batch_size = 34
epochs = 15
history = model.fit(x_train_s ,
        y_train_s ,
        batch_size = batch_size ,
        epochs = epochs ,
        validation_split = 0.1)

print("predictions")
predictions = model.predict ( x_test_s )


#  Prikazi test accuracy i matricu zabune
scores = model.evaluate( x_test_s , y_test_s , verbose =0 )
print("Test loss", scores[0])
print("Test accuracy:", scores[1])

y_test_s = np.argmax(y_test_s, axis=1)
predi = np.argmax(predictions, axis=1)#sto ovo radi?

confusionMatrix = ConfusionMatrixDisplay(confusion_matrix(y_test_s, predi))
confusionMatrix.plot()
plt.show()

# spremi model
model.save("model.keras")
