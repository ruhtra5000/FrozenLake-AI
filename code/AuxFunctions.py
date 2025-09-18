import gymnasium as gym
import threading
from GeneticManager import GeneticManager 

# Class with some auxiliar functions
class AuxFunctions:
    def __init__(self, env: gym.Env, geneticManager: GeneticManager):
        self.env = env
        self.geneticManager = geneticManager

    # Calculates fitness for all individuals
    def calculateFullPopulationFitness(self, numThreads = 2):
        fitnessCalculation = [[] for _ in range(numThreads)]
        threads = []
        elementsPerThread = self.geneticManager.populationLen // numThreads

        # Create threads
        for i in range(numThreads):
            t = threading.Thread(
                target=self.subPopulationFitness, 
                args=(self.geneticManager.population[i*elementsPerThread:(i+1)*elementsPerThread], fitnessCalculation, i)
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
    def subPopulationFitness(self, subpopulation: list, results: list, index: int):
        results[index] = [{"ind": ind, "fitness": self.geneticManager.fitness(ind, self.env)} for ind in subpopulation]

    # Calculates average fitness by generation
    def avgFitness(self, population):
        totalFitness = 0.0
        for ind in population:
            totalFitness += ind["fitness"]
        
        return totalFitness/self.geneticManager.populationLen

    # Visual test with the best individual found
    def testBestIndividual(self, bestIndividual):
        testEnv = gym.make("FrozenLake-v1", is_slippery=True, render_mode = "human")

        print(f"\n\nBest individual data\nGeneration: {bestIndividual["gen"]}\nDNA: {bestIndividual["ind"]}\nFitness: {bestIndividual["fitness"]}")

        for _ in range(10):
            state, _ = testEnv.reset()

            for _ in range(self.geneticManager.maxStepsPerEpisode):
                newState, _ , terminated, truncated, _ = testEnv.step(bestIndividual["ind"][state])
                state = newState

                if terminated or truncated:
                    break