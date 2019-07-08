#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 15:37:40 2018

"""

import csv
from pylab import *

def linearfit(X, Y):
    W =dot(inv(dot(X.T, X)),dot(X.T,Y))
    return W


def main():
    rfile = 'student_debt.csv'
    csvfile = open(rfile, 'rt')
    data = csv.reader(csvfile, delimiter = ',')
    X = []
    Y = []
    for i, row in enumerate(data):
        X.append(float(row[0]))
        Y.append(float(row[1]))
    X = array(X)
    X_temp = X[:]
    Y = array(Y)
    one = ones((len(X)))
    X = row_stack((one,X))
    X = X.T
    Y = reshape(Y, (len(Y),1))
    weight = linearfit(X,Y)
    print (weight)

    it = np.arange(2004, 2050, 1)
    plt.ylabel('Debt')
    plt.xlabel('Time')
    plt.title('Linear regression')
    g = weight[1] * it + weight[0]
    plt.plot(it, g, 'k')
    plt.plot(X_temp, Y, 'bo')
    plt.show()
    print (weight[1] * 2050 + weight[0])
main()