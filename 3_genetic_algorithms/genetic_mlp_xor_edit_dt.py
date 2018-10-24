import numpy as np
from random import uniform, randint, sample
from operator import itemgetter, attrgetter, methodcaller
#from matplotlib import pyplot as plt
#nepoch = 20;
#training set
datasetxor = np.array([
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0],
]);
inf = 1000000000.0;
lrate = 1.5;#learning rate
nepoch = 5;#number of iterations
population = [];
for i in range(10):
    pop_list = [];
    for j in range(13):
        pop_list.append(uniform(-5, 5));
    population.append(pop_list);
    #population.append([uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5)]);

def sigmoid(val):
    res = 1/(1+np.exp(-val));
    return res;

def get_fitness(wts, lrate, nepoch):
    fitness = 0.0;
    sq_error = 0.0;
    acc = 0.0;
    tempacc = 0.0;
    for i in range(nepoch):
        sq_error = 0.0;
        tempacc = 0.0;
        for row in datasetxor:
            delwts = [];
            for k in range(len(wts)):
                delwts.append(0);
            #forward propagation
            y1 = sigmoid(wts[0]+wts[1]*row[0]+wts[2]*row[1]);
            y2 = sigmoid(wts[3]+wts[4]*row[0]+wts[5]*row[1]);
            y3 = sigmoid(wts[6]+wts[7]*row[0]+wts[8]*row[1]);
            y4 = sigmoid(wts[9]+wts[10]*y1+wts[11]*y2+wts[12]*y3);
            y4p = 0;
            if y4>=0.5:
                y4p = 1;
            if y4p==row[-1]:
                tempacc += 1.0;
            #error backpropagation
            e4 = row[-1]-y4;
            sq_error += e4*e4;
            del4 = y4*(1.0-y4)*e4;
            delwts[9] = -lrate*del4;
            delwts[10] = lrate*y1*del4;
            delwts[11] = lrate*y2*del4;
            delwts[12] = lrate*y3*del4;
            del3 = y3*(1.0-y3)*del4*wts[12];
            delwts[6] = -lrate*del3;
            delwts[7] = lrate*row[0]*del3;
            delwts[8] = lrate*row[1]*del3;
            del2 = y2*(1.0-y2)*del4*wts[11];
            delwts[3] = -lrate*del2;
            delwts[4] = lrate*row[0]*del2;
            delwts[5] = lrate*row[1]*del2;
            del1 = y1*(1.0-y1)*del4*wts[10];
            delwts[0] = -lrate*del1;
            delwts[1] = lrate*row[0]*del1;
            delwts[2] = lrate*row[1]*del1;
            for k in range(len(wts)): #weight updation at last
                wts[k] += delwts[k];
    
        tempacc /= len(datasetxor);
        if tempacc>acc:
            acc = tempacc;
        
    if sq_error==0.0:
        fitness = inf;
    else:
        fitness = 1/sq_error;
    return fitness, acc;
    
def genetic_algo(population, lrate, nepoch):
    for i in range(3, 0, -1): #to generate 3->2->1 for 2^3->2^2->2^1 for crossover of genes
        #print();
        #print("Current population of weights: ", population);
        #sort by fitness
        fitness = [];
        for wts in population:
            fit, acc = get_fitness(wts, lrate, nepoch);
            fitness.append((wts, fit)); #get fitness for 5 epoch
            sorted(fitness, key=itemgetter(1), reverse=True); #sort by decreasing order of fitness    
        cross_num = 2**i; #for 2^i
        newpop = [];
        #print(len(fitness));
        for j in range(0, cross_num, 2): #loop with step size 2
            #print(fitness[j]);
            parent1 = fitness[j][0];
            parent2 = fitness[j+1][0];
            shuffle_num = randint(1, len(parent1)-1);
            cross1 = sample(parent1, shuffle_num);
            cross2 = sample(parent2, shuffle_num);
            p1left = [];
            p1right = [];
            p2left = [];
            p2right = [];
            for ele in parent1:
                if ele in cross1:
                    p1left.append(ele);
                else:
                    p1right.append(ele);
                    
            for ele in parent2:
                if ele in cross2:
                    p2left.append(ele);
                else:
                    p2right.append(ele);
            
            newparent1 = [];
            newparent2 = [];
            for ele in p2left:
                newparent1.append(ele);
            for ele in p1right:
                newparent1.append(ele);
            for ele in p1left:
                newparent2.append(ele);
            for ele in p2right:
                newparent2.append(ele);
            
            newpop.append(newparent1);
            newpop.append(newparent2);
        
        for j in range(cross_num, len(population)):
            newpop.append(fitness[j][0]);
        
        mutation_num = randint(0, cross_num-1);
        mut_wt = randint(0, len(newpop[mutation_num])-1);
        newpop[mutation_num][mut_wt] += uniform(0, 1); #random mutation between 0 to 1 inclusive
        for j in range(len(newpop)):
            population[j] = [];
            for ele in newpop[j]:
                population[j].append(ele);
                    
    fit, acc = get_fitness(population[0], lrate, 1); #for 1 epoch run the trained model and check accuracy
    #print();
    print("Updated weights after cross-over and mutation: ", population[0]);
    print("Accuracy of prediction of xor after cross-over and mutation is: ", acc*100);
    return acc;
    
#training and testing the perceptron    
    #[[uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5)], [uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5)], [uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5)], [uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5)], [uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5), uniform(-5, 5)]];
#print(population, sample(population[0], 5));
#choose population and crossover using genetic algorithm
tempacc = 0.0;
acc = 0.0;
iterations = 5;
for i in range(iterations):
    print();
    print("For iteration "+str(i+1)+":");
    #print("Current population of weights: ", population);
    tempacc = genetic_algo(population, lrate, nepoch);
    if tempacc>acc:
        acc = tempacc;
        
#maximum prediction accuracy
print();
print("Maximum Accuracy of prediction of xor after "+str(iterations)+" iteration of genetic algorithm with Multilayer Perceptron (MLP) is: ", acc*100);