import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


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
                    label=cl)


# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age", "EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None)
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " +
      "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " +
          "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()


# KNN METODA
KNN = KNeighborsClassifier(n_neighbors=5)
KNN.fit(X_train_n, y_train)

# Evaluacija modela KNN
y_test_p_KNN = KNN.predict(X_test_n)
y_train_p = KNN.predict(X_train_n)
print("knn: ")
print("Tocnost train: " +
      "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " +
      "{:0.3f}".format((accuracy_score(y_test, y_test_p_KNN))))

# granica odluke pomocu KNN
plot_decision_regions(X_train_n, y_train, classifier=KNN)
plt.title("Tocnost: " +
          "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

KNN2 = KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(1, 25)}
knn_gscv = GridSearchCV(KNN2, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
knn_gscv.fit(X_train_n, y_train)
print("najoptimalnija vrijednost:", knn_gscv.best_params_)
print(knn_gscv.best_score_)

# unakrsna validacija
scores = cross_val_score(KNN2, X_train_n, y_train, cv=5)
print('cv_scores mean:{}'.format(np.mean(scores)))


# 1.1.
# za razliku od logisticke regresije, knn ima bolju tocnost(i sa skupom testiranja i treniranja),
#  granica vise nije ravna linija vec zakrivljena i bolje odvaja podatke

# 1.2
# razlika kada upotrijebimo K=1 i K=100: granica s 1 izgleda vrlo ne definirano i raspisano, postoji puno "otoka"(tocnost je 0.994-dogada se overfitting)
# sa K=100  granica jedne klase je jako povucena u drugu klasu, vidi se iz samog grafa (tocnost je 0.787-dogada se underfitting)
