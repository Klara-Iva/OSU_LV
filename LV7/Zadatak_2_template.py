import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs\\test_3.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w, h, d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

# lakat metdom da se vidi koliko ima boja i onda za svaku sliku to napravit
# binarno prebaci boje u 1 ili 0, pa ce bit ili ne biti


clusters = 4
# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
(h, w, c) = img.shape
img_array = np.reshape(img, (h*w, c))

# rezultatna slika
img_array_aprox = img_array.copy()

# broj razlicitih boja u slici
print("postoji ", len(np.unique(img_array_aprox, axis=0)), "razlicith boja u slici")

# primjena K-means algoritma
km = KMeans(n_clusters=clusters, n_init="auto")
labels = km.fit_predict(img_array_aprox)

# promjena vrijednosti u K-means centre
rgb_cols = km.cluster_centers_.astype(np.float64)
print(rgb_cols)
img_quant = np.reshape(rgb_cols[labels], (h, w, c))

plt.imshow(img_quant)
plt.show()

# binarna slika s obzirom na klasu
for i in range(clusters):
    bit_values = labels == [i]
    binary_img = np.reshape(bit_values, (img.shape[0:2]))
    binary_img = binary_img*1
    x = int(i/2)
    y = i % 2
    plt.imshow(binary_img)
    plt.show()
