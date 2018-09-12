# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 10:32:49 2018

@author: kirito

"""
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

def unroll(X):
    image_unrolled = []
    for row in X:
        row_unrolled = []
        for pixel in row:
            pixel_unrolled = [pixel[1]]
            #print(pixel[1])
            row_unrolled = row_unrolled + pixel_unrolled
            #print(row_unrolled)
        image_unrolled = image_unrolled + row_unrolled
    #print(image_unrolled)
    return image_unrolled

#TRAINING SET
X = []
Y = []
trainpath = 'river/'
ext = '.png'
filename = 'river'
uri = trainpath + filename + ext
img = mpimg.imread(uri)
X = unroll(img)
#print(X)

#reduce X size
X = X[0::64]
print(len(X))
#TESTING SET
X_test = []
Y_test = [] #actual label of test images
test_path = 'river/'
filename = 'test64'
ext = '.png'
uri = test_path + filename + ext
img = mpimg.imread(uri)
X_test = unroll(img)
print(len(X_test))


#clf = MLPClassifier(batch_size=1,solver='lbfgs',activation='logistic',learning_rate='constant', hidden_layer_sizes=(5,5))
#clf.fit(X, Y) 
#Y_predicted = clf.predict(X_test)
#print(clf.predict_proba(X_test))

#HUMAN TESTING
# Change i value to change input and see output.
#input_image_uri = test_path + str(i) + ext
#plt.subplot(121)
#plt.imshow(mpimg.imread(input_image_uri))
#plt.title('Input Image')
#plt.subplot(122)
#input_image_class = clf.predict([unroll(mpimg.imread(input_image_uri))])
#plt.imshow(mpimg.imread(path + str(input_image_class[0]) + ext))
#plt.title('Output Image')
#plt.show()
