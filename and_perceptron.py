# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 10:02:04 2018

@author: Student
"""
import numpy as np
from matplotlib import pyplot as plot

X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
Y = np.array([0, 0, 0, 1])

X = X.T
w = np.zeros(len(X[0]))
b = 1
alpha = 0.1
flag = 0
for i in range(100):
    j = i%4
    z = np.dot(w,X[j])+b
    if z>=0:
        a = 1
    else:
        a = 0
    e = Y[j]-a
    if e==0:
        flag = flag+1
    else:
        flag = 0
    if flag==4:
        print("Total iterations of the stochastic gradient are: ", i)
        break
    w = w + alpha*e*X[j]
    b = b + alpha*e

print(w)
print(b)

for i in range(4):
    z = np.dot(w,X[i])+b
    if z>=0:
        a = 1
        plot.plot(X[i][0],X[i][1], 'ro')
    else:
        a = 0
        plot.plot(X[i][0],X[i][1], 'bo')
    #print(X[i], a)
        
k = np.arange(0, 2, 0.2)
plot.plot(k, -1*(k*w[0]/w[1] + b/w[1]), 'g--')