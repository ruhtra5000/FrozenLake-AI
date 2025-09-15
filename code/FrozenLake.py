import gymnasium as gym
from GeneticManager import GeneticManager

# Creating environment and some variables
env = gym.make("FrozenLake-v1", render_mode = "human")

geneticManager = GeneticManager(env.action_space)

numberGenerations = 10
stepsPerGeneration = 10 

# "Main"
for generation in range(numberGenerations):
    env.reset()
    dna = geneticManager.generateChromosome()
    for step in range(stepsPerGeneration):
        newState, reward, terminated, truncated, info = env.step(int(dna[step]))
