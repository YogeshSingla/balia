# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 12:06:21 2018

@author: STUDENT1
"""

from sklearn.neural_network import MLPClassifier
import numpy as np
import scipy as sp
import imageio
im= imageio.imread("river64.png")
imageio.imwrite('river-gray.png', im[:, :, 0])
datasetX=imageio.imread("river-gray.png")
datasetY = np.zeros((64,64))

for i in range(64):
    for j in range(64):
        if (datasetX[i][j] > 75):
            datasetY[i][j] = 1
        else:
            datasetY[i][j] = 0

datasetX = np.reshape(datasetX, [64 * 64, 1])
print(datasetX)
print(datasetY)
datasetY = np.reshape(datasetY, [64 * 64, 1])
y=datasetY.ravel()
print(np.shape(y))
mlp = MLPClassifier(solver='lbfgs', alpha=0.001,activation='logistic',
                        hidden_layer_sizes=(5, 5), max_iter=1000)
mlp.fit(datasetX, y)
im= imageio.imread("test64.png")
imageio.imwrite('test64-gray.png', im[:, :, 0])
datasetX=imageio.imread("test64-gray.png")
datasetX = np.reshape(datasetX, [64 * 64, 1])

print(np.shape(datasetX))
outputY=[]
y_pred = mlp.predict(datasetX)
print(y_pred)
for predic in y_pred:
        if (predic >= 0.5):
            outputY.append(255)
        else:
            outputY.append(0)
outputY = np.reshape(outputY, [64, 64])
print("Output image saved.")
sp.misc.imsave('output64.png', outputY)

    