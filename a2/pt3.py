import random
import operator
import math
import numpy
from deap import  base,creator,tools,gp,algorithms

fp3 = 'ass2_data/part3/regression'

def geneticFunction():
    # Constants
    generation = 0
    max_generations = 1000
    matingPB = .8
    mutatePB = .4
    nTopScores = 3
    topScoresPerGen = []

    # Init population
    train = Trainer3()
    toolbox = train.toolbox
    pop = toolbox.populate(n=300)
    scores = [toolbox.evaluate(p)[0] for p in pop]
    topScoresPerGen.append(sorted(zip(scores, pop), key=lambda x: x[0])[:nTopScores])
    
    # Run through generations
    while min(scores) > 0 and generation < max_generations:
        generation += 1

        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))

        # Mate and mutate random selections of the offspring
        for c1,c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < matingPB:
                toolbox.mate(c1, c2)
        for o in offspring:
            if random.random() < mutatePB:
                toolbox.mutate(o)
        
        # Shallow copy offspring and re-evaluate scores
        pop[:] = offspring
        scores = [toolbox.evaluate(p)[0] for p in pop]

        # Record the best scores and print the generation results
        topScoresPerGen.append(sorted(zip(scores, pop), key=lambda x: x[0])[:nTopScores])
        print(f"Gen: {generation}\n\tMin: {min(scores)}, Max: {max(scores)}, Avg:{sum(scores) / len(scores)}")
        
    # Print best scores across the generations
    bestScores = sorted([item for sublist in topScoresPerGen for item in sublist], key=lambda x: x[0])[:nTopScores]
    for score in bestScores:
        print(score[0], ": ", str(score[1]))

class Trainer3:
    # Class varaibles
    toolbox = None
    domain = None
    vals = []

    def protectedDiv(self, x, y):
        # Used to make sure iterations don't throw an error
        try:
            return x / y
        except ZeroDivisionError:
            return 1

    def grade(self, expr):
        errs = 0
        func = self.toolbox.compile(expr=expr)
        for x, y in self.vals:
            if not (func(x)-.05 <= y <= func(x)+.05):
                errs += 1
        return errs / len(self.vals),

    def __init__(self):
        # Parse data
        data = open(fp3,'r')
        data.readline()

        for line in data.readlines():
            x, y = line.split()
            self.vals.append((float(x), float(y)))

        # Construct the domain for solving the problem
        domain = gp.PrimitiveSet('main', 1)
        domain.addPrimitive(operator.add, 2)
        domain.addPrimitive(operator.sub, 2)
        domain.addPrimitive(operator.mul, 2)
        domain.addPrimitive(self.protectedDiv, 2)
        domain.addEphemeralConstant("rand", lambda: random.randint(1,5))

        # Use a fitness that minimizes error
        creator.create('FitnessMin', base.Fitness, weights=(-1.,))
        creator.create("Instance", gp.PrimitiveTree, fitness=creator.FitnessMin)

        # Register all the functions we need
        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=domain, min_=1, max_=3)
        toolbox.register("instance", tools.initIterate, creator.Instance, toolbox.expr)
        toolbox.register("populate", tools.initRepeat, list, toolbox.instance)
        toolbox.register("compile", gp.compile, pset=domain)
        toolbox.register("evaluate", self.grade)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mutate", gp.genFull, min_=1, max_=3)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mutate, pset=domain)
        toolbox.register("select", tools.selTournament, tournsize=3)
        # Prevents trees from getting too tall
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

        self.domain = domain
        self.toolbox = toolbox

def run():
    geneticFunction()
    
if __name__ == '__main__':
    run()