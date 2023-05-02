import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report

labels = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}


def plot_decision_regions(X, y, classifier, resolution=0.02):

    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor='w',
                    label=labels[cl])


# ucitaj podatke
df = pd.read_csv("penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'].replace({'Adelie': 0,
                       'Chinstrap': 1,
                       'Gentoo': 2}, inplace=True)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm', 'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()
y = y[:, 0]  # od više matrice se stvara jedna

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)

# a) koliko unikatnih primjera postoji
train_values, train_count = np.unique(y_train, return_counts=True)
test_values, test_count = np.unique(y_test, return_counts=True)
ind = np.arange(3)
width = 0.4

plt.bar(train_values, train_count, color='blue', width=0.4, label='Train')
plt.bar(test_values+0.4, test_count, color='red', width=0.4, label='Test')
plt.xticks(ind + width / 2, ('Adelie', 'Chinstrap', 'Gentoo'))
plt.legend(loc='upper right')
plt.show()


# b
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# c Pronadite u atributima izgradenog modela parametre modela
t0 = log_reg.intercept_[0]
print(t0)
print(log_reg.coef_)


# d Poziv plot_decision_region
plot_decision_regions(X_train, y_train, log_reg)

# e Matrica zabune, tocnost i metrike

y_predict = log_reg.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
print("Matrica:", cm)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_predict))
disp.plot()
print("Tocnost:", accuracy_score(y_test, y_predict))
print(classification_report(y_test, y_predict))
plt.show()


# f Dodati još jednu ulazni velicinu i napraviti klasifikaciju podataka na skupu za testiranje
output_variable = ['species']
input_variables = ['bill_length_mm',
                   'flipper_length_mm',
                   'body_mass_g']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()
y = y[:, 0]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)


LogisticRegression_model2 = LogisticRegression()
log_reg = LogisticRegression_model2.fit(X_train, y_train)
y_predict2 = LogisticRegression_model2.predict(X_test)

print("Tocnost:", accuracy_score(y_test, y_predict2))
print(classification_report(y_test, y_predict2))
# rezultati klasifikacije se smanjuju s porastom ulaznih velicina
