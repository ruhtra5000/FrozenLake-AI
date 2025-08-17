
# Class responsible for all genetic procedures
class GeneticManager:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    # Generates a random chromosome
    def generateChromosome(self):  
        chromosome = ""
        for _ in range(10):
            chromosome += str(self.actionSpace.sample())

        return chromosome