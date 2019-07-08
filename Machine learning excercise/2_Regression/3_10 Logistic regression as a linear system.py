#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 22:19:02 2018

@author: caoweidong
"""

import csv
from pylab import *

def linearfit(X, Y):
    W =dot(inv(dot(X.T, X)),dot(X.T,Y))
    return W

def sigmoid(x):
    return 1 / (1 + exp(-x))

def main():
    rfile = 'bacteria_data.csv'
    csvfile = open(rfile, 'rt')
    data = csv.reader(csvfile, delimiter = ',')
    X = []
    Y = []
    for i, row in enumerate(data):
        X.append(float(row[0]))
        Y.append(float(row[1]))
    X = array(X)
    Y = array(Y)
    X1 = X[:]
    Y1 = Y[:]
    one = ones(len(X))
    X = column_stack((one,X))
    Y = Y.T
    Y = log(Y/(1-Y))
    w = linearfit(X,Y)
    print(w)

    it = np.arange(0,26,1)
    plt.ylabel('bacteria data')
    plt.xlabel('Time')
    plt.title('Logistic regression')
    g = sigmoid(w[1] * it + w[0])
    plt.plot(it, g, 'k')
    plt.plot(X1, Y1, 'bo')
    plt.show()
main()