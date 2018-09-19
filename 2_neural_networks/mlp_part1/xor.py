# -*- coding: utf-8 -*-
"""
Created on Wed Sep 05 12:01:07 2018

@author: STUDENT1
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 05 10:14:41 2018

@author: yogesh singla
Code was written by reference from :
    http://scikit-learn.org/stable/modules/neural_networks_supervised.html

The classifier used can be found at :
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/neural_network/multilayer_perceptron.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

X = [[0,0],[0,1],[1,0],[1,1]]
Y = [0,1,1,0]
X_test = [[0,0],[0,1],[1,0],[1,1]]
Y_test = [0,1,1,0]
colors = [0, 1, 1, 0]

plt.xlabel('input 1')
plt.ylabel('intput 2')
plt.scatter([0, 0, 1, 1 ],[0, 1, 0, 1],c=colors)
plt.legend()
plt.show()



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
clf = MLPClassifier(solver='lbfgs',activation='logistic',learning_rate='constant', hidden_layer_sizes=(5,2), random_state=1)
clf.fit(X, Y) 
Y_predicted = clf.predict(X_test)

correct_predictions = 0
total = len(Y_test)
for predicted,actual in zip(Y_predicted,Y_test):
    print(predicted,actual)
    if predicted == actual :
        correct_predictions = correct_predictions + 1
    
print("Accuracy : %.4f " %(float(correct_predictions)/total))