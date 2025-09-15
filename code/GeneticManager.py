import gymnasium as gym
import random

# Class responsible for all genetic procedures
class GeneticManager:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace
        self.populationLen = 150 # Number of individuals
        self.population = self.initialPopulation()
        self.tournamentSize = 3
        self.episodesPerFitness = 30
        self.maxStepsPerEpisode = 100
        self.crossoverRate = 0.8
        self.mutationRate = 0.05        

    # Generates the first population (full random)
    def initialPopulation(self):
        return [self.generateRandomChromosome() for _ in range(self.populationLen)]

    # Generates a random chromosome
    def generateRandomChromosome(self):  
        chromosome = []
        for _ in range(16):
            chromosome.append(self.actionSpace.sample()) 

        return chromosome
    
    # Tournament selection
    def tournament(self, env: gym.Env):
        bestIndividuals = []
        index = 0
        bestFitness = 0.0
        bestIndex = 0
        
        while index < self.populationLen:
            # Gets a group of individuals
            selectedIndividuals = self.population[index:(index+self.tournamentSize)]
            
            # Seeks best individual per group
            fitnessValues = [self.fitness(ind, env) for ind in selectedIndividuals]

            maxFitness = max(fitnessValues) 
            maxFitnessIndex = fitnessValues.index(maxFitness)

            bestIndividuals.append(self.population[(index+maxFitnessIndex)])

            # Seeks best individual in general
            if maxFitness > bestFitness:
                bestFitness = maxFitness
                bestIndex = index+maxFitnessIndex

            # Updates index (going to the next group)
            index += self.tournamentSize

        print("\n\nBest individual in generation:")
        print(f"Fitness: {bestFitness} | Index: {bestIndex}")
        print(self.population[bestIndex])

    # Fitness function
    def fitness(self, dna: list, env: gym.Env):
        totalReward = 0.0
        for _ in range(self.episodesPerFitness):
            state, _ = env.reset()
            episodeReward = 0.0

            for _ in range(self.maxStepsPerEpisode):
                newState, reward, terminated, truncated, _ = env.step(dna[state])
                episodeReward += reward
                state = newState

                if terminated or truncated:
                    break

            totalReward += episodeReward

        #Returns average reward
        return totalReward/self.maxStepsPerEpisode