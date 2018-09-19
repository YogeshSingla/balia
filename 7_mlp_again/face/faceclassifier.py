# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 10:32:49 2018

@author: kirito

Tips:
    Use tab to autocomplete.

Tested in Spyder (Python 2.7)
"""
import matplotlib.image as mpimg
from sklearn.neural_network import MLPClassifier

path = 'TrainDatabase/'
ext = '.jpg'
uri = path + str(1) + ext
img = mpimg.imread(uri)
import matplotlib.pyplot as plt

def unroll(X):
    image_unrolled = []
    for row in X:
        row_unrolled = []
        for pixel in row:
            pixel_unrolled = [sum([x for x in pixel])/ (3.0*255)]
            row_unrolled = row_unrolled + pixel_unrolled
        image_unrolled = image_unrolled + row_unrolled
    return image_unrolled

#TRAINING SET
X = []
Y = []
for i in range(1,21):
    uri = path + str(i) + ext
    img = mpimg.imread(uri)
    X.append(unroll(img))
    Y.append(int((i+1)/2))

#TESTING SET
X_test = []
Y_test = [] #actual label of test images
test_path = 'TestDatabase/'
for i in range(1,11):
    uri = test_path + str(i) + ext
    img = mpimg.imread(uri)
    X_test.append(unroll(img))
    Y_test.append(i)

print(len(X[0]))
#print(X)
clf = MLPClassifier(batch_size=1,max_iter=1000,solver='lbfgs',activation='relu',learning_rate='constant', hidden_layer_sizes=(5,5,5,5))
clf.fit(X, Y) 
Y_predicted = clf.predict(X_test)
#print(clf.predict_proba(X_test))

correct_predictions = 0
total = len(Y_test)
for predicted,actual in zip(Y_predicted,Y_test):
    print(actual,predicted)
    if predicted == actual :
        correct_predictions = correct_predictions + 1
    
print("\nAutomated testing")
print("Accuracy : %.4f " %(float(correct_predictions)/total))

#HUMAN TESTING
# Change i value to change input and see output.
print("\nManual Testing (looped)")
for i in range(1,11):
    input_image_uri = test_path + str(i) + ext
    plt.subplot(121)
    plt.imshow(mpimg.imread(input_image_uri))
    plt.title('Input Image')
    plt.subplot(122)
    input_image_class = clf.predict([unroll(mpimg.imread(input_image_uri))])
    plt.imshow(mpimg.imread(path + str(input_image_class[0]) + ext))
    plt.title('Output Image')
    plt.show()
