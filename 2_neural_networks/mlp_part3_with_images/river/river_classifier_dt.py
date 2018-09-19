import numpy as np
#import scipy as sp
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
#from sklearn.preprocessing import normalize

#dataset = [];
x_train = [];
x_test = [];
y_train = [];
y_test = [];
y_pred = [];

def get_image(file, x):
    img = Image.open(file).convert('L');
    x = np.array(img);
    #print(np.shape(x));
    #print(np.shape(x_train));
    x = np.reshape(x, (len(x)*len(x[0]), 1));
    #print(x);
    return x;
    
def process_image(x, y):
    for i in range(len(x)):
        if x[i]>245:
            y.append(1);
        else:
            y.append(0);            
    return y;
            
def train_data(x_train, y_train):
    nepoch = 1000;
    #activation = 'tanh';
    activation = 'logistic';
    mlp = MLPClassifier(solver='lbfgs', alpha=0.001, activation=activation, hidden_layer_sizes=(5, 2), max_iter=nepoch, random_state=None);
    mlp.fit(x_train, y_train);
    return mlp;

def pred_test_data(x_test, y_test, y_pred, mlp):
    y_pred = mlp.predict(x_test);
    acc = accuracy_score(y_test, y_pred, normalize=False)/len(y_test);
    print("Test accuracy: ", acc);
    print(y_pred);
    return y_pred;
    
def create_image_file(y_pred):
    y_img = [];
    for i in range(len(y_pred)):
        if y_pred[i]>=0.5:
            y_img.append(255);
        else:
            y_img.append(0);
    y_img = np.array(y_img);
    #y_img = np.reshape(y_img, (64, 64));#size of original test image file
    # Use PIL to create an image from the new array of pixels
    print("Ouput image file saved successfully");
    img_size = (64, 64);
    new_image = Image.new('L', img_size);
    new_image.putdata(y_img);
    new_image.save('river_pred.png');
    #pix = np.array(Image.open('river_pred.png').convert('L'));
    #print(pix);
    
#input processing
ip1 = input("Enter input image file path: ");
ip2 = input("Enter test image file path: ");
x_train = get_image(ip1, x_train);
y_train = process_image(x_train, y_train);
mlp = train_data(x_train, y_train);
x_test = get_image(ip2, x_test);
y_test = process_image(x_test, y_test);
y_pred = pred_test_data(x_test, y_test, y_pred, mlp);
create_image_file(y_pred);
