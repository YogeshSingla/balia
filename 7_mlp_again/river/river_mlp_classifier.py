# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 10:32:49 2018

@author: kirito
 
Assignment 5: 
    Binary Classification of river and non-river using neural network:
        Take river64.png image as input and read the all pixel value of grayscale image. 
        First convert the each pixel in two categories if pixel value is greater than 245 make it 1 and rest 0.
            if (datasetX[i][j] > 245):
                datasetY[i][j] = 1
            else:
                datasetY[i][j] = 0
        Apply neural network learning with 
        learning rate (alpha=.001), 
        Gradient Descent optimization function and 
        run for 1000 epoch. 
        Classify in river and non-river category and return the image with white for river (pixel value = 0) and black for rest of the class (pixel value = 255).
        Refer the logic.
            if (predic >= 0.5):
                outputY.append(255)
            else:
                outputY.append(0)
"""
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from skimage.transform import rescale

def unroll(X):
    image_unrolled = []
    for row in X:
        row_unrolled = []
        for pixel in row:
            
            pixel_unrolled = [sum([x for x in pixel[0:3]])/3] # since it's black and white image
            row_unrolled = row_unrolled + pixel_unrolled
        image_unrolled = image_unrolled + row_unrolled
    return image_unrolled

#RESCALE TO 64x64
original_image_path = 'river/'
ext = '.png'
filename = 'river'
uri = original_image_path + filename + ext
img = mpimg.imread(uri)
img_rescaled = rescale(img, 1.0 / 8.0)
test_file = 'rescaled_input'
trainpath = 'river/'
mpimg.imsave(trainpath+test_file+ext,img_rescaled)

#TRAINING SET
X = []
Y = []
X_train = []
Y_train = []
train_img = img_rescaled
X = unroll(train_img)
threshold = (245.0-0.0)/(255.0-0.0)
for i in X:
    if i < threshold:
        Y.append(0)
    else:
        Y.append(1)
    X_train.append([i])
Y_train = Y
#sanity check
import numpy as np
train_image_output = np.reshape(Y,(64,64))
#print(train_image_output)
mpimg.imsave("river/train_output_image.png",train_image_output,cmap='gray')

#TESTING SET
X_test = []
Y_test = [] #actual label of test images
test_path = 'river/'
filename = 'test64'
ext = '.png'
uri = test_path + filename + ext
img = mpimg.imread(uri)
X_test_tmp = unroll(img)
for i in X_test_tmp:
    X_test.append([i])
#print(len(X_test))
#print(len(Y))


clf = MLPClassifier(alpha=0.001,solver='lbfgs',activation='logistic',learning_rate='constant', hidden_layer_sizes=(5,5),max_iter=1000)
clf.fit(X_train, Y_train) 
Y_predicted_probability = clf.predict_proba(X_test)
print((Y_predicted_probability))
for i in Y_predicted_probability:
    if i[1] > 0.05105:
        Y_test.append(1)
    else:
        Y_test.append(0)
test_image_output = np.reshape(Y_test,(64,64))
print(Y_test)
mpimg.imsave("river/test_image_output.png",test_image_output,cmap='gray')

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
