import gymnasium as gym
from GeneticManager import GeneticManager
from operator import itemgetter

# Creating environment and some variables
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode = None)

geneticManager = GeneticManager(env.action_space)

numberGenerations = 100

# Calculates average fitness by generation
def avgFitness(population):
    totalFitness = 0.0
    for ind in population:
        totalFitness += ind["fitness"]
    
    return totalFitness/geneticManager.populationLen

# Visual test with the best individual found
def testBestIndividual(bestIndividual):
    testEnv = gym.make("FrozenLake-v1", is_slippery=True, render_mode = "human")

    for _ in range(5):
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
        popWithFit = geneticManager.calculateFullPopulationFitness(env)
        sortedPop = sorted(popWithFit, key=itemgetter('fitness'), reverse=True)
        
        # Get best individuals (elitism)
        numIndividuals = int(geneticManager.populationLen / 10)
        elite = sortedPop[0:numIndividuals]

        # Prints (monitoring)
        print(f"\nBest in gen {generation}:\n{elite[0]}")
        print(f"Avg. Fitness: {avgFitness(popWithFit)}")

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