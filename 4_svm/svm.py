# -*- coding: utf-8 -*-
"""
@author: yogesh singla
Code was written by reference from :
    http://axon.cs.byu.edu/Dan/678/miscellaneous/SVM.example.pdf

"""

import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

#linearly separable data
X_n = [[1,0],[0,1],[-1,0],[0,-1]]
X_p = [[3,2],[6,2],[3,-2],[6,-2]]

#non-linear data
X_n = [[2,2],[2,-2],[-2,2],[-2,-2]]
X_p = [[-1,0],[0,-1],[0,1],[1,0]]

colors = [0, 0, 0, 0,1,1,1,1]
X_all = X_n + X_p
X_all_nl = X_n_nl + X_p_nl
y = [0,0,0,0,1,1,1,1]
x1 = [x[0] for x in X_all]
x2 = [x[1] for x in X_all]
x1_nl = [x[0] for x in X_all_nl]
x2_nl = [x[1] for x in X_all_nl]

plt.xlabel('x1')
plt.ylabel('x2')
plt.scatter(x1,x2,c=colors)
plt.legend()
plt.title("Dataset Linearly Separable")
plt.show()


plt.xlabel('x1')
plt.ylabel('x2')
plt.scatter(x1_nl,x2_nl,c=colors)
plt.legend()
plt.title("Dataset Non-Linearly Separable")
plt.show()


from sklearn import svm
clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3,gamma = 0.1, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf.fit(X_all,y)

print(clf.predict([[0.,0.]]))

plot_decision_regions(X=np.asarray(X_all), 
                      y=np.asarray(y),
                      clf=clf, 
                      legend=2)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title('SVM Decision Region Boundary')

"""
Parameters:	

C : float, optional (default=1.0)

    Penalty parameter C of the error term.
kernel : string, optional (default=’rbf’)

    Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’
    , ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples).
degree : int, optional (default=3)

    Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
gamma : float, optional (default=’auto’)

    Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.

    Current default is ‘auto’ which uses 1 / n_features, if gamma='scale' is passed then it uses 1 / (n_features * X.std()) as value of gamma. The current default of gamma, ‘auto’, will change to ‘scale’ in version 0.22. ‘auto_deprecated’, a deprecated version of ‘auto’ is used as a default indicating that no explicit value of gamma was passed.
coef0 : float, optional (default=0.0)

    Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
shrinking : boolean, optional (default=True)

    Whether to use the shrinking heuristic.
probability : boolean, optional (default=False)

    Whether to enable probability estimates. This must be enabled prior to calling fit, and will slow down that method.
tol : float, optional (default=1e-3)

    Tolerance for stopping criterion.
cache_size : float, optional

    Specify the size of the kernel cache (in MB).
class_weight : {dict, ‘balanced’}, optional

    Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are supposed to have weight one. The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))
verbose : bool, default: False

    Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm that, if enabled, may not work properly in a multithreaded context.
max_iter : int, optional (default=-1)

    Hard limit on iterations within solver, or -1 for no limit.
decision_function_shape : ‘ovo’, ‘ovr’, default=’ovr’

    Whether to return a one-vs-rest (‘ovr’) decision function of shape (n_samples, n_classes) as all other classifiers, or the original one-vs-one (‘ovo’) decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2). However, one-vs-one (‘ovo’) is always used as multi-class strategy.

    Changed in version 0.19: decision_function_shape is ‘ovr’ by default.

    New in version 0.17: decision_function_shape=’ovr’ is recommended.

    Changed in version 0.17: Deprecated decision_function_shape=’ovo’ and None.
random_state : int, RandomState instance or None, optional (default=None)

    The seed of the pseudo random number generator used when shuffling the data for probability estimates. If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.

Attributes:	

support_ : array-like, shape = [n_SV]

    Indices of support vectors.
support_vectors_ : array-like, shape = [n_SV, n_features]

    Support vectors.
n_support_ : array-like, dtype=int32, shape = [n_class]

    Number of support vectors for each class.
dual_coef_ : array, shape = [n_class-1, n_SV]

    Coefficients of the support vector in the decision function. For multiclass, coefficient for all 1-vs-1 classifiers. The layout of the coefficients in the multiclass case is somewhat non-trivial. See the section about multi-class classification in the SVM section of the User Guide for details.
coef_ : array, shape = [n_class * (n_class-1) / 2, n_features]

    Weights assigned to the features (coefficients in the primal problem). This is only available in the case of a linear kernel.

    coef_ is a readonly property derived from dual_coef_ and support_vectors_.
intercept_ : array, shape = [n_class * (n_class-1) / 2]

    Constants in decision function.

"""
