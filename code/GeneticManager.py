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
        self.mutationRate = 0.15

    # Generates the first population (full random)
    def initialPopulation(self):
        return [self.generateRandomChromosome() for _ in range(self.populationLen)]

    # Generates a random chromosome
    def generateRandomChromosome(self):  
        chromosome = []
        for _ in range(16):
            chromosome.append(int(self.actionSpace.sample())) 

        return chromosome
    
    # Calculates fitness for all individuals
    def calculateFullPopulationFitness(self, env: gym.Env):
        return [{"ind": ind, "fitness": self.fitness(ind, env)} for ind in self.population]
    
    # Tournament selection
    def tournament(self, group: list):
        # Group contain dicts:
        # {"ind": list (dna), "fitness": float}

        # Select random individuals
        subgroup = random.sample(group, k=self.tournamentSize)

        # Return best individual (max fitness)
        return max(subgroup, key= lambda x: x["fitness"])

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
        return totalReward/self.episodesPerFitness
    
    # Crossover two individuals (generate 2 new individuals)
    def crossover(self, parent1: list, parent2: list):
        probCrossover = random.random()
        # Do the crossover
        if probCrossover < self.crossoverRate:
            cutIndex = random.randint(1, len(parent1)-1) # Random cut point
    
            child1 = parent1[0:cutIndex] + parent2[cutIndex:]
            child2 = parent2[0:cutIndex] + parent1[cutIndex:]

            return child1, child2
        # Returns the parents
        else:
            return parent1, parent2

    # Mutate chromosome
    def mutation(self, dna: list):
        for i in range(len(dna)):
            if random.random() < self.mutationRate:
                allActions = [0, 1, 2, 3]
                allActions.remove(dna[i])
                dna[i] = random.choice(allActions)
        
        return dna
