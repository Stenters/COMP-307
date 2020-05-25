import random
import operator
import math
import numpy
from deap import  base,creator,tools,gp,algorithms

fp4 = ('ass2_data/part4/training.txt', 'ass2_data/part4/test.txt')


def geneticClassification():
    """
    Part 4
        classify data set using GP
            195 instances; 75 anomaly; 120 normal (36 features [25-175])
        create data files (training.txt & test.txt)
        Report
            terminal set
            function set
            Fitness function (desc in plain language && math function)
            Parameter values & stopping criteria
            Desciribe considerations creating test and training split
            classification accuracy (over 10 random runs) on training & test
            3 best programs & fitness values
            analyse one of the best programs to identify paterns that you can find and why it can solve the problem
    """
    # Constants
    generation = 0
    max_generations = 50
    matingPB = .5
    mutatePB = .1
    nTopScores = 3
    topScoresPerGen = []

    # Init population
    train = Trainer4()
    toolbox = train.toolbox
    pop = toolbox.populate(n=300)
    scores = [toolbox.evaluate(p)[0] for p in pop]
    topScoresPerGen.append(sorted(zip(scores, pop), key=lambda x: x[0])[:nTopScores])
    
    # Run through generations
    while min(scores) > 0 and generation < max_generations:
        generation += 1

        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))

        for c1,c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < matingPB:
                toolbox.mate(c1, c2)
        for o in offspring:
            if random.random() < mutatePB:
                toolbox.mutate(o)
        
        pop[:] = offspring
        scores = [toolbox.evaluate(p)[0] for p in pop]

        topScoresPerGen.append(sorted(zip(scores, pop), key=lambda x: x[0])[:nTopScores])
        print(f"Gen: {generation}\n\tMin: {min(scores)}, Max: {max(scores)}, Avg:{sum(scores) / len(scores)}")
        
    for score in topScoresPerGen[-1]:
        print(score[0], ": ", str(score[1]))


class Trainer4:
    toolbox = base.Toolbox()
    domain = gp.PrimitiveSet('main', 1)
    trainingSet = []
    testSet = []

    def gradeOnTrainingSet(self, expr):
        errs = 0
        func = self.toolbox.compile(expr=expr)
        for x, y in self.trainingSet:
            if (func(*x) != y):
                errs += 1
        return errs / len(self.trainingSet),

    def gradeOnTestSet(self, expr):
        errs = 0
        func = self.toolbox.compile(expr=expr)
        for x, y in self.testSet:
            if (func(*x) != y):
                errs += 1
        return errs / len(self.testSet),

    def parseFiles(self, test, training):
        for line in training.readlines():
            line = line.split()
            classification = line[-1] == "'Anomaly'"
            features = list(map(int,line[:-1]))
            self.trainingSet.append((features, classification))
        
        for line in test.readlines():
            line = line.split()
            classification = line[-1] == "'Anomaly'"
            features = list(map(int,line[:-1]))
            self.testSet.append((features, classification))

    def __init__(self):
        self.parseFiles(open(fp4[0], 'r'), open(fp4[1], 'r'))

        domain = gp.PrimitiveSet('main', 36)
        domain.addPrimitive(operator.lt, 2)
        domain.addPrimitive(operator.le, 2)
        domain.addPrimitive(operator.eq, 2)
        domain.addPrimitive(operator.ne, 2)
        domain.addPrimitive(operator.ge, 2)
        domain.addPrimitive(operator.gt, 2)

        creator.create('FitnessMin', base.Fitness, weights=(-1.,))
        creator.create("Instance", gp.PrimitiveTree, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=domain, min_=1, max_=5)
        toolbox.register("instance", tools.initIterate, creator.Instance, toolbox.expr)
        toolbox.register("populate", tools.initRepeat, list, toolbox.instance)
        toolbox.register("compile", gp.compile, pset=domain)
        toolbox.register("evaluate", self.gradeOnTrainingSet)
        toolbox.register("test", self.gradeOnTestSet)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mutate", gp.genFull, min_=1, max_=5)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mutate, pset=domain)
        toolbox.register("select", tools.selTournament, tournsize=3)
        # Prevents trees from getting too tall
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

        self.domain = domain
        self.toolbox = toolbox


def run():
    geneticClassification()
    
if __name__ == '__main__':
    run()