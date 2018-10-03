import numpy as np
from matplotlib import pyplot as plt
nepoch = 20;
#training set
datasetxor = np.array([
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0],
]);

def sigmoid(val):
    res = 1/(1+np.exp(-val));
    return res;
    
def predict_xor(row, w):
    activation = w[0];#for bias
    for i in range(len(row)):
        activation += w[i+1]*row[i];
        
    activation = sigmoid(activation);
    #print(activation);
    if activation>=0.5:
        res = 1;
    else:
        res = 0;
    return activation, res;
    
def train_perceptron(dataset, lrate, nepoch):
    w1 = np.zeros(len(dataset[0]));#bias is w[0]
    w2 = np.zeros(len(dataset[0]));#for hidden layers
    w3 = np.zeros(len(dataset[0]));#for output layer
    y = np.zeros(3);#y0 and y1 for hidden layer and y2 for ouput layer
    delta = np.zeros(3);#for three gradients
    
    for epoch in range(nepoch):
        for row in dataset:
            #forward propagation
            y[0], y0b = predict_xor(row[:2], w1);
            y[1], y1b = predict_xor(row[:2], w2);
            y[2], y2b = predict_xor(y[:2], w3);
            error = row[-1]-y[2];
            delta[2] = y[2]*(1-y[2])*error;
            w3[0] -= lrate*delta[2];#updating bias
            for i in range(len(w3)-1):
                w3[i+1] += lrate*y[i]*delta[2];
            delta[1] = y[1]*(1-y[1])*delta[2]*w3[2];
            w2[0] -= lrate*delta[1];
            for i in range(len(w2)-1):
                w2[i+1] += lrate*row[i]*delta[1];
            delta[0] = y[0]*(1-y[0])*delta[2]*w3[1];
            w1[0] -= lrate*delta[0];
            for i in range(len(w1)-1):
                w1[i+1] += lrate*row[i]*delta[0];
    
    w = np.append(w1, w2, axis=0);#to append column wise
    w = np.append(w, w3, axis=0);
    return w;
 
def predict(x, w):
    w1 = w[:3];#bias is w[0]
    w2 = w[3:6];#for hidden layers
    w3 = w[6:];#for output layer
    y = np.zeros(3);
    y[0], y0b = predict_xor(x, w1);
    y[1], y1b = predict_xor(x, w2);
    y[2], res = predict_xor(y[:2], w3);
    print(y[2]);
    return res;

def plot_graph_xor(w):
    x_intercept = -w[0]/w[1];#for x1 on x-axis
    y_intercept = -w[0]/w[2];#for x2
    x_cord = [x_intercept, 0];
    y_cord = [0, y_intercept];
    plt.plot(x_cord, y_cord);
    plt.plot(0, 0, 'bo');
    plt.plot(0, 1, 'bo');
    plt.plot(1, 0, 'bo');
    plt.plot(1, 1, 'ro');
    plt.xlabel('x1');
    plt.ylabel('x2');
    plt.title('Classification for XOR gate');
    
#input processing
ip = input("Enter x1 and x2 values for xor gate: ");
x = [int(y) for y in ip.split(' ')];

#training and testing the perceptron    
lrate = 0.1;#learning rate
nepoch = 100000;#number of iterations
w = train_perceptron(datasetxor, lrate, nepoch);
#print(w);
bias = np.array([w[0], w[3], w[6]]);
#print(w[0]);
#print(w[3]);
#print(w[6]);
print(bias);
pred = predict(x, w);
print("Predicted XOR output for given input: ", pred);
#plot_graph_xor(w);
