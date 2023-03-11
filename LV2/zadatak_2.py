import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
data = np.loadtxt('data.csv', delimiter=',', skiprows=1, dtype=float)

size = len(data)
print('Broj osoba:', size)
size = int(size)

x = np.array(data[:, 1])
y = np.array(data[:, 2])
plt.title('all persons')
plt.xlabel('height')
plt.ylabel('weight')
plt.scatter(x, y, c='g', linewidths=1, s=0.1)
plt.show()

x1 = x[::50]
y1 = y[::50]
plt.title('every 50th person')
plt.xlabel('height')
plt.ylabel('weight')
plt.scatter(x1, y1, c='r', linewidths=1, s=0.5)
plt.show()

print('Min height: ', x.min())
print('Max height: ', x.max())
print('Average height: ', x.mean())

male = (data[:, 0] == 1)
female = (data[:, 0] == 0)
print('Min male height: ', data[male, 1].min())
print('Max male height: ', data[male, 1].max())
print('Average male height: ', data[male, 1].mean())
print('Min female height: ', data[female, 1].min())
print('Max female height: ', data[female, 1].max())
print('Average female height: ', data[female, 1].mean())
