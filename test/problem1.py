import numpy

"""
problem 1
"""

def value_constraint(val,past_val):
   
    if(val>4):
        val = past_val
    elif val<-4:
        val = past_val
    else:
        val = val
    return val

def calc_fitness(X, chromosome):
    fitness = numpy.sum(chromosome*X, axis=1)
    return fitness

def selection(pop, fitness, num_parents):
    parents = numpy.empty((num_parents, pop.shape[1])) #initialise parents matrix
    for parent_num in range(num_parents):
        max_fitness_index = numpy.where(fitness == numpy.max(fitness))
        max_fitness_index = max_fitness_index[0][0]
        parents[parent_num, :] = pop[max_fitness_index, :]
        fitness[max_fitness_index] = -99999999999 #to remove this for next iteration
    return parents

def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    crossover_point = numpy.uint8(offspring_size[1]/2)+1
    for k in range(offspring_size[0]):
        parent1_idx = k%parents.shape[0]
        parent2_idx = (k+1)%parents.shape[0]
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring):
    k = numpy.random.randint(0,6,1)[0]
    for i in range(offspring.shape[0]):
        random_value = numpy.random.uniform(-1.0, 1.0, 1)
        past_val = offspring[i,k]
        offspring[i, k] = offspring[i, k] + random_value
        offspring[i,k] = value_constraint(offspring[i,k],past_val)
    return offspring

weights = [4,-2,7,5,11,1]
num_weights = len(weights)
population = 8
num_parents_mating = 4
pop_size = (population,num_weights)
initial_population = numpy.random.uniform(low=-4.0, high=4.0, size=pop_size)
new_population = initial_population

print("Initial population:")
print(new_population)
plot_x = []
plot_y = []
num_generations = 150
for generation in range(num_generations):
    fitness = calc_fitness(weights, new_population)
    parents = selection(new_population, fitness,num_parents_mating)
    offspring = crossover(parents,offspring_size=(pop_size[0]-parents.shape[0], num_weights))
    offspring = mutation(offspring)
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring
    print("\nGeneration #"+str(generation)+ "\t Fitness: "+str(numpy.max(numpy.sum(new_population*weights, axis=1))))
    plot_y.append(numpy.max(numpy.sum(new_population*weights, axis=1)))
    plot_x.append(generation)

fitness = calc_fitness(weights, new_population)
best_match_index = numpy.where(fitness == numpy.max(fitness))

print("\n\nFinal chromosome : "+"\n"+str(new_population[best_match_index, :]))
print("\nFinal chromosome fitness : "+str(fitness[best_match_index]))

import matplotlib.pyplot as plt
plt.plot(plot_x,plot_y)
plt.axhline(y=120,color='r',linewidth=1)
plt.xlabel("iteration")
plt.ylabel("fitness")
plt.title("Genetic Algorithm ")