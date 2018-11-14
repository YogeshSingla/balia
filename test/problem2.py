# -*- coding: utf-8 -*-
"""
Created on Wed Sep 05 10:14:41 2018

@author: yogesh singla
Code was written by reference from :
    http://scikit-learn.org/stable/modules/neural_networks_supervised.html

The classifier used can be found at :
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/neural_network/multilayer_perceptron.py
"""

import csv #for importing csv data file

#Read the CSV file into the python environment
data_list = []
with open('iris/iris.csv', 'rt') as csvfile:   
    read_obj = csv.reader(csvfile, delimiter = ',')
    for row in read_obj:
        data_list.append(row)
    field_headings = data_list[0]
    data_list.remove(data_list[0])

#print raw data
print(field_headings)

#convert data from string to float
for i in range(len(data_list)):
    for j in range(4):
        data_list[i][j] = float(data_list[i][j])

def numify(a):
    if a == 'Iris-virginica':
        return 1
    if a == 'Iris-versicolor':
        return 2
    if a == 'Iris-setosa':
        return 3

def norm_min_max(X):
    a = 0
    b = 1
    X_min = min(X)
    X_max = max(X)
    if(X_min == X_max):
        return X
    X_minmax = []
    for i in X:
        norm_x = a + ((i - X_min) * (b - a)) / (X_max - X_min)
        X_minmax.append(norm_x)
    return X_minmax

col1 = [x[0] for x in data_list]
col2 = [x[1] for x in data_list]
col3 = [x[2] for x in data_list]
col4 = [x[3] for x in data_list]
col5 = [x[4] for x in data_list]

col1 = norm_min_max(col1)
col2 = norm_min_max(col2)
col3 = norm_min_max(col3)
col4 = norm_min_max(col4)

data_list = []
for c1,c2,c3,c4,c5 in zip(col1,col2,col3,col4,col5):
    data_list.append([c1,c2,c3,c4,c5])

import numpy as np
import matplotlib.pyplot as plt
x0 = [x0[0] for x0 in data_list] #sepal length
x0_label = 'sepal length'
x1 = [x1[1] for x1 in data_list] #sepal width
x1_label = 'sepal width'
x2 = [x2[2] for x2 in data_list] #petal length
x2_label = 'petal length'
x3 = [x3[3] for x3 in data_list] #petal width
x3_label = 'petal width'
colors = [numify(c[4]) for c in data_list]
#area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

labels = [x0_label, x1_label, x2_label, x3_label]
columns = [x0 , x1, x2, x3]

#all 12 plots of two attributes at a time
for i in range(len(labels)):
    for j in range(i,len(labels)):
        if(i != j):
            plt.xlabel(labels[i])
            plt.ylabel(labels[j])
            plt.scatter(columns[i],columns[j],c=colors)
            plt.legend()
            plt.show()
        
        
from sklearn.neural_network import MLPClassifier

#training set of 35 samples from each class
X1 = [x[0:4] for x in data_list[0:35]]
X2 = [x[0:4] for x in data_list[50:85]]
X3 = [x[0:4] for x in data_list[100:135]]
X = X1 + X2 + X3

Y1 = [numify(x[4]) for x in data_list[0:35]]
Y2 = [numify(x[4]) for x in data_list[50:85]]
Y3 = [numify(x[4]) for x in data_list[100:135]]
Y = Y1 + Y2 + Y3

#testing set of 15 samples unseen from each class
X1 = [x[0:4] for x in data_list[35:50]]
X2 = [x[0:4] for x in data_list[85:100]]
X3 = [x[0:4] for x in data_list[135:150]]
X_test = X1 + X2 + X3
Y1 = [numify(x[4]) for x in data_list[35:50]]
Y2 = [numify(x[4]) for x in data_list[85:100]]
Y3 = [numify(x[4]) for x in data_list[135:150]]
Y_test = Y1 + Y2 + Y3

"""
Reference for MLPClassifier
activation : {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default ‘relu’

    Activation function for the hidden layer.

        ‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x
        ‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
        ‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
        ‘relu’, the rectified linear unit function, returns f(x) = max(0, x)

solver : {‘lbfgs’, ‘sgd’, ‘adam’}, default ‘adam’

    The solver for weight optimization.

        ‘lbfgs’ is an optimizer in the family of quasi-Newton methods.
        ‘sgd’ refers to stochastic gradient descent.
        ‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba

learning_rate : {‘constant’, ‘invscaling’, ‘adaptive’}, default ‘constant’

    Learning rate schedule for weight updates.

        ‘constant’ is a constant learning rate given by ‘learning_rate_init’.
        ‘invscaling’ gradually decreases the learning rate learning_rate_ at each time step ‘t’ using an inverse scaling exponent of ‘power_t’. effective_learning_rate = learning_rate_init / pow(t, power_t)
        ‘adaptive’ keeps the learning rate constant to ‘learning_rate_init’ as long as training loss keeps decreasing. Each time two consecutive epochs fail to decrease training loss by at least tol, or fail to increase validation score by at least tol if ‘early_stopping’ is on, the current learning rate is divided by 5.

    Only used when solver='sgd'.

"""
clf = MLPClassifier(solver='lbfgs',activation='logistic',learning_rate='constant', hidden_layer_sizes=(2), random_state=1)
clf.fit(X, Y) 
Y_predicted = clf.predict(X_test)

correct_predictions = 0
total = len(Y_test)
for predicted,actual in zip(Y_predicted,Y_test):
    print(predicted,actual)
    if predicted == actual :
        correct_predictions = correct_predictions + 1
    
print("Accuracy : %.4f " %(float(correct_predictions)/total))