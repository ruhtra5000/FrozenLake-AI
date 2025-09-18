import gymnasium as gym
from operator import itemgetter
from GeneticManager import GeneticManager
from AuxFunctions import AuxFunctions

# Creating environment and some variables
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode = None)

geneticManager = GeneticManager(env.action_space)
auxFunctions = AuxFunctions(env, geneticManager)

numberGenerations = 200

# "Main"
def main():
    bestIndividual = {"gen": 0, "ind": ["None"], "fitness": 0.0}

    for generation in range(numberGenerations):
        # Calculate fitness 
        popWithFit = auxFunctions.calculateFullPopulationFitness()
        sortedPop = sorted(popWithFit, key=itemgetter('fitness'), reverse=True)
        
        # Get best individuals (elitism)
        numIndividuals = int(geneticManager.populationLen / 10)
        elite = sortedPop[0:numIndividuals]

        # Prints (monitoring)
        print(f"\nBest in gen {generation+1}:\n{elite[0]}")
        print(f"Avg. Fitness: {auxFunctions.avgFitness(popWithFit):.4f}")

        # Storaging best individual (overall)
        if bestIndividual["fitness"] < elite[0]["fitness"]:
            bestIndividual = {
                "gen": generation+1, 
                "ind": elite[0]["ind"], 
                "fitness": elite[0]["fitness"]
            }

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

    auxFunctions.testBestIndividual(bestIndividual)

main()