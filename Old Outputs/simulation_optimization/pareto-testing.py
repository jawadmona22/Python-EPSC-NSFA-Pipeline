import random
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# Objective functions
def objective1(x):
    return x**2

def objective2(x):
    return (x - 2)**2

# Define the problem
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))  # Minimize both f1 and f2
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Initialization function
def create_individual():
    return [random.uniform(-5, 5)]  # Initialize individuals with values in range [-5, 5]

# Evaluation function
def evaluate(individual):
    x = individual[0]
    f1 = objective1(x)
    f2 = objective2(x)
    return f1, f2  # Return both objective values

# Set up the toolbox
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Crossover operator
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=0.2)  # Mutation operator
toolbox.register("select", tools.selNSGA2)  # Selection operator (NSGA-II)

# Create the initial population
population = toolbox.population(n=100)  # Population size of 100 individuals

# Run the evolutionary algorithm
num_generations = 100
crossover_prob = 0.7
mutation_prob = 0.2

for gen in range(num_generations):
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < crossover_prob:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < mutation_prob:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid_individuals))
    for ind, fit in zip(invalid_individuals, fitnesses):
        ind.fitness.values = fit

    population[:] = offspring

# Extract Pareto Front
pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

# Plot Pareto Front
f1_values = [ind.fitness.values[0] for ind in pareto_front]
f2_values = [ind.fitness.values[1] for ind in pareto_front]

plt.scatter(f1_values, f2_values)
plt.xlabel("Objective 1 (f1)")
plt.ylabel("Objective 2 (f2)")
plt.title("Pareto Front")
plt.show()
