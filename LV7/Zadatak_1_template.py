import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering


def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X, y = make_blobs(n_samples=n_samples, random_state=random_state)

    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                          centers=4,
                          cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                          random_state=random_state)
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)

    # 2 grupe
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)

    else:
        X = []

    return X


# generiranje podatkovnih primjera
X = generate_data(500, 1)
# prikazi primjere u obliku dijagrama rasprsenja-bez boja samo obicni generirani podaci
plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri')
plt.show()


X = generate_data(500, 1)
# ako imamo gotove podatke->X = data.iloc[:, :-1].values


# Kmeans metoda za odredivanje centroida
km = KMeans(n_clusters=3, init='random', n_init=5, random_state=0)
km.fit(X)
labels = km.predict(X)


# prikazi primjere u obliku dijagrama rasprsenja
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[
            :, 1], s=250, marker='*', c='red', label='Centroids')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.title('podatkovni primjeri')
plt.show()
# za vise grupa, mogao bi se koristit i kod:
# plt.scatter(X[labels == 0, 0], X[labels == 0, 1], s=50, c='lightgreen',  label='Grupa 1')
# plt.scatter(X[labels == 1, 0], X[labels == 1, 1], s=50, c='orange',  label='Grupa 2')
# plt.scatter(X[labels == 2, 0], X[labels == 2, 1], s=50, c='lightblue', label='Grupa 3')


# Lakat metoda za odredivanje najboljeg K- trebala bi se prvo ona obavit a onda scatteri
distortions = []
# veci range da bude bolje prikazano sto je najbolje upotrijebit
K = range(1, 15)
for k in K:
    kmm = KMeans(n_clusters=k, init='random', n_init=5, random_state=0)
    kmm.fit(X)
    distortions.append(kmm.inertia_)

plt.figure()
plt.plot(K, distortions)
plt.show()


# tocnost KMeans mozemo izracunati prema:
# y_true = data.iloc[:, -1].values
# accuracy = np.mean(y_true == labels)
# print("tocnost je:" ,accuracy)


# 1.1 postoji 3 grupe
# 1.2  promjenom broja K stvara se onoliko grupa koliki je K
#
