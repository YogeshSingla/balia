# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 22:32:25 2018

@author: Arghya
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 05 10:11:53 2018

@author: STUDENT1
"""
from itertools import product
import numpy as np
from numpy import exp, array, random, dot
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize 
#downloading of dataset and setting headers
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Species']
iris = datasets.load_iris()
X = iris.data[:, 2:4]
y=iris.target
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

#initial statistics
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()


#using standard scaling on sepa-length,sepal-width ,petal -length,petal-width and encoding on 
#different species of iris

x =iris.data[:,0:4]
y =iris.target
X_normalized=normalize(x,axis=0)


#train test split 70-30
x_train, x_test, y_train, y_test =train_test_split(X_normalized,y,test_size=0.30)
activation = ['identity', 'logistic', 'tanh', 'relu']
#model building hidden layer 1=5 neurons and hiddenlayer 2=2 neurons
for act in activation:
        mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,activation=act,
                            hidden_layer_sizes=(5, 2), random_state=1)
        mlp.fit(x_train, y_train)
        
        #prediction results
        y_pred = mlp.predict(x_test)
        print('Activation Function Used: '+act)
        print('Confusion Matrix')
        print(confusion_matrix(y_test, y_pred))
        target_names=['Iris-setosa','Iris-versicolor','Iris-virginica']
        print('Classification Report')
        print(classification_report(y_test, y_pred, target_names=target_names))

#visualization of decision boundary
X = iris.data[:, [0, 2]]
y = iris.target
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,activation='logistic',
                            hidden_layer_sizes=(5, 2), random_state=1)
mlp.fit(X, y)
#prediction results

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(sharex='col', sharey='row', figsize=(10, 8))



Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

axarr.contourf(xx, yy, Z, alpha=0.4)
axarr.scatter(X[:, 0], X[:, 1], c=y,
                              s=20, edgecolor='k')
axarr.set_title('MLP using Gradient Descent Optimizer ,2 hidden layer and sigmoid activation')

plt.show()

