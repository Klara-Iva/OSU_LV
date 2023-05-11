import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt


# ucitaj CIFAR-10 podatkovni skup
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# prikazi 9 slika iz skupa za ucenje
plt.figure()
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.xticks([]),plt.yticks([])
    plt.imshow(X_train[i])

plt.show()


# pripremi podatke (skaliraj ih na raspon [0,1]])
X_train_n = X_train.astype('float32')/ 255.0
X_test_n = X_test.astype('float32')/ 255.0

# 1-od-K kodiranje
y_train = to_categorical(y_train, dtype ="uint8")
y_test = to_categorical(y_test, dtype ="uint8")

#ako koristimo bas bazu onda dijelimo na x i y i ako bude error da mismatcha (1,) i (3,) ide ovaj kod: za kategoricki klase
#X = data[["length (cm)","width (cm)","length (m)","width (m)"]].to_numpy()
#y = data["izlazni"].to_numpy().reshape(-1, 1)
#encoder = OneHotEncoder()
#y = encoder.fit_transform(y).toarray()


# CNN mreza
model = keras.Sequential()
model.add(layers.Input(shape=(32,32,3)))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

# definiraj listu s funkcijama povratnog poziva
my_callbacks = [
    
    keras.callbacks.EarlyStopping(monitor='val_loss',
    patience = 5 ,
    verbose = 1 ),
    keras.callbacks.TensorBoard(log_dir = 'logs/cnn_dropoutAndEarlyStopping',
                                update_freq = 100)
]

model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model.fit(X_train_n,
            y_train,
            epochs = 5,
            batch_size = 256,
            callbacks = my_callbacks,
            validation_split = 0.1)


score = model.evaluate(X_test_n, y_test, verbose=0)
print('Tocnost na testnom skupu podataka:',score[1])

#b
#dodavanjem dropout sloja, smanjuje se tocnost na testnom skupu podataka, sto je veci broj dropouta to je manja tocnost
#dodavanjem i povecanjem dropoout sloja, accurany i val_acurrancy imaju sve blizu vrijednost

#c
#tocnost se dodavanjem funkcije povratnog poziva za rano zaustavljanje sada nalazi ispod tocnosti mreze bez te funckije i bez dropouta,
#i iznad tocnosti mreze sa samo dropoutom


#d

#d.1 povecanjem velicine batcha dolazi do manjih iteracija s manjom tocnosti i vecim loss, u suprotnom-za vrlo malene batcheve treba vise epoha i vrlo dugo je trajanej jedne
#d.2
#s jako malom stopom ucenja gotovo ne dolazi do promjene u lossu i accucnarcyu, kod jako velikoe vriejnosti stope ucenja  je loss vlro velik a accurnacy malen
#d.3
#izbacivanjem slojeva iz mreze, smanjuje se vrijeme potrebno za svaku epohu ali se smanuje i tocnost na testnom skupu
#d.4 vrijeme se smanjilo za oko polovicu pocetnog vremena, ali isto se odnosi i na tocnost koja se nije bas prepolovila ali je blizu
