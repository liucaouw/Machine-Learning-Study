from __future__ import division
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import pylab


# load data
def load_data():
    data = np.array(np.genfromtxt('knn_data.csv', delimiter=','))
    x = np.reshape(data[:, 0], (np.size(data[:, 0]), 1))
    y = np.reshape(data[:, 1], (np.size(data[:, 1]), 1))
    for i in np.arange(len(data)):
        if data[i][2] == 0:
            data[i][2] = -1
    return data, x, y

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

def knn(data, x, y, k):
    sum = 0
    m = np.zeros((len(data), 2))
    for i in np.arange(len(data)):
        m[i][0] = (x - data[i][0])**2 + (y - data[i][1])**2
        m[i][1] = data[i][2]
    m = m.tolist()
    m.sort(key=lambda x: x[0])
    m = np.asarray(m)
    for i in range(0, k):
        sum=sum+ m[i][1]
    ynew = sign(sum)
    return ynew

data, x, y = load_data()
N = 12000
x1 = np.random.rand(N) * 10
y1 = np.random.rand(N) * 10


for i in np.arange(len(x1)):
    # call the knn function and set the value of k
    k=1
    ynew = knn(data, x1[i], y1[i], k)
    if ynew >= 0:
        plt.scatter(x1[i], y1[i], color='red')
    else:
        plt.scatter(x1[i], y1[i], color='purple')


plt.plot(x, y, 'bo')
plt.xlim(0.0, 10.0)
plt.ylim(0.0, 10.0)
plt.axis('off')
plt.show()

