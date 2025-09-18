import gymnasium as gym
from GeneticManager import GeneticManager
from operator import itemgetter
import threading

# Creating environment and some variables
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode = None)

geneticManager = GeneticManager(env.action_space)

numberGenerations = 200

# Calculates fitness for all individuals
def calculateFullPopulationFitness(env: gym.Env, numThreads = 2):
    fitnessCalculation = [[] for _ in range(numThreads)]
    threads = []
    elementsPerThread = geneticManager.populationLen // numThreads

    # Create threads
    for i in range(numThreads):
        t = threading.Thread(
            target=subPopulationFitness, 
            args=(env, geneticManager.population[i*elementsPerThread:(i+1)*elementsPerThread], fitnessCalculation, i)
        )

        threads.append(t)
        t.start() # initialize a thread
        
    # Wait for all threads to end
    for t in threads:  
        t.join()

    # Join the results
    finalFitness = []
    for i in range(numThreads):
        finalFitness = finalFitness + fitnessCalculation[i]

    return finalFitness

# Calculates fitness for a portion of the population (Thread target function)
def subPopulationFitness(env: gym.Env, subpopulation: list, results: list, index: int):
    results[index] = [{"ind": ind, "fitness": geneticManager.fitness(ind, env)} for ind in subpopulation]

# Calculates average fitness by generation
def avgFitness(population):
    totalFitness = 0.0
    for ind in population:
        totalFitness += ind["fitness"]
    
    return totalFitness/geneticManager.populationLen

# Visual test with the best individual found
def testBestIndividual(bestIndividual):
    testEnv = gym.make("FrozenLake-v1", is_slippery=True, render_mode = "human")

    print(f"\n\nBest individual data\nDNA: {bestIndividual["ind"]}\nFitness: {bestIndividual["fitness"]}")

    for _ in range(10):
        state, _ = testEnv.reset()

        for _ in range(geneticManager.maxStepsPerEpisode):
            newState, reward, terminated, truncated, _ = testEnv.step(bestIndividual["ind"][state])
            state = newState

            if terminated or truncated:
                break

# "Main"
def main():
    bestIndividual = {"ind": [], "fitness": 0.0}

    for generation in range(numberGenerations):
        # Calculate fitness 
        popWithFit = calculateFullPopulationFitness(env)
        sortedPop = sorted(popWithFit, key=itemgetter('fitness'), reverse=True)
        
        # Get best individuals (elitism)
        numIndividuals = int(geneticManager.populationLen / 10)
        elite = sortedPop[0:numIndividuals]

        # Prints (monitoring)
        print(f"\nBest in gen {generation+1}:\n{elite[0]}")
        print(f"Avg. Fitness: {avgFitness(popWithFit):.4f}")

        # Storaging best individual (overall)
        if bestIndividual["fitness"] < elite[0]["fitness"]:
            bestIndividual = elite[0]

        # Generate new individuals
        newPopulation = [ind["ind"] for ind in elite]
        
        while len(newPopulation) < geneticManager.populationLen:
            parent1 = geneticManager.tournament(elite)["ind"]
            parent2 = geneticManager.tournament(elite)["ind"]

            child1, child2 = geneticManager.crossover(parent1, parent2)

            child1 = geneticManager.mutation(child1)
            child2 = geneticManager.mutation(child2)

            newPopulation.append(child1)
            newPopulation.append(child2)

        geneticManager.population = newPopulation

    testBestIndividual(bestIndividual)

main()