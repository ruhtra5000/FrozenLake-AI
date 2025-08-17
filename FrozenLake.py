import gymnasium as gym

# Creating environment and some variables
env = gym.make("FrozenLake-v1", render_mode = "human")

# Generates a random chromosome
def generateChromosome():  
    chromosome = ""
    for _ in range(10):
        chromosome += str(env.action_space.sample())

    return chromosome


generations = 10
stepsPerGeneration = 10 

# "Main"
for gnrt in range(generations):
    env.reset()
    dna = generateChromosome()
    for step in range(stepsPerGeneration):
        newState, reward, terminated, truncated, info = env.step(int(dna[step]))
