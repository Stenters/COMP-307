'''
TODO
    Part1
    Part2
    Part3
    Part4
    readme
    Report
'''

# Packages
import random
import operator
import math
import numpy
from sklearn.neural_network import MLPClassifier
from deap import  base,creator,tools,gp,algorithms

# File Paths
fp1 = 'ass2_data/part1/dataset'
fp2 = ('ass2_data/part2/wine', 'ass2_data/part2/wine_test', 'ass2_data/part2/wine_training')
fp3 = 'ass2_data/part3/regression'
fp4 = ('ass2_data/part4/training.txt', 'ass2_data/part4/test.txt')


# Part 1


def perceptron():
    """
    Part 1
        Train a Perceptron to classify data 
        Create data file
        Report
            Report classification accuracy after 200 epochs
            Analyzse limitations for not achieving better results
    """
    epochs = 10
    net_err = 0
    weights = [0,0,0]
    learnRate = .1
    bias = 0
    data = []

    datafile = open(fp1, 'r')
    datafile.readline()

    for line in datafile.readlines():
        data.append(((int(line[0]),int(line[2]),int(line[4])), int(line[6])))
    
    while (epochs > 0):
        epochs -= 1
        iterations = 0
        error = 0
        
        for line in data:
            if predict(line[0], line[1], weights, bias, learnRate) != line[1]:
                error += 1
        
            iterations += 1
        
        error /= iterations
        net_err += error
        
    print(f"\n\tfinal bias: {bias}\n\tfinal weights: {weights}\n\tfinal error: {net_err / 10}")


def predict(data, classification, weights, bias, learnRate):
    sum = bias

    for i in range(0,len(data)):
        sum += data[i] * weights[i]

    sum = 1 if sum >= 0 else 0

    bias += learnRate * (classification - sum)

    for i in range(0, len(data)):
        weights[i] += learnRate * (classification - sum) * data[i]

    return sum


### Part 2


def packagedNeuralNetwork():
    """
    Part 2
        Train a neural network to classify data using an existing package
            sklearn
        Reformat data for package
        Report
            Determine the network architecture (# inoput nodes, # output, # hidden nodes (assume one layer)). Describe rationale
            Determine learing parameters (inc learning rate, momentum, initial weight ranges, etc). Describe rationale
            Determine termination criteria. Describe rationale
            Report reults (10 independent experiments with different random seeds) on training and test. Analyse results and make conclusions
            Compare vs nearest neighbor
            Describe why you used that package
    """
    testFeatures, testClasses = parseFile(open(fp2[1], 'r')) 
    trainingFeatures, trainingClasses = parseFile(open(fp2[2], 'r'))
    classifier = MLPClassifier(solver='sgd', hidden_layer_sizes=(1,13), random_state=1, verbose=False)

    for i in range(1, len(trainingFeatures)-1):
        classifier.fit(trainingFeatures[:i], trainingClasses[:i])
        print(classifier.predict(testFeatures))

    # classifier.fit(trainingFeatures, trainingClasses)
    # res = classifier.predict(testFeatures)

    # error = 0
    # iters = 0
    # for i in range(len(res) - 1):
    #     if (testClasses[i] != res[i]):
    #         error += 1
    #     iters += 1
    # print(res)
    # print(testClasses)
    # print(f"err: {error / iters}")


def parseFile(file):
    features = []
    classes = []
    file.readline()
    for line in file.readlines():
        vals = line.split()
        vals = [float(x) for x in vals]
        features.append(vals[:-1])
        classes.append(vals[-1])
    return features, classes

### Part 3


def geneticFunction():
    """
    Part 3
        Use genetic programming to generate a mathematic function to relate inputs and outputs
        Use existing package
            DEAP?
        
        Report
            Terminal set
            Function set
            Fitness function (desc in plain language && math function)
            Parameter values & stopping criteria
            Mean squared error for 10 random runs and avg value
            3 diff programs and their fitness values
            Analyse one of the best programs and explain why it can solve the problem in the task
    """

    # constants
    generation = 0
    max_generation = 200
    matingPB = .5
    mutatePB = .1

    # Init stats
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    # Init population
    train = Trainer3()
    toolbox = train.toolbox
    pop = toolbox.populate(n=300)
    hof = tools.HallOfFame(3)

    algorithms.eaSimple(pop, toolbox, matingPB, mutatePB, max_generation, stats=mstats, verbose=True, halloffame=hof)
    for h in hof:
        print(h)
    

class Trainer3:
    toolbox = base.Toolbox()
    domain = gp.PrimitiveSet('main', 1)
    vals = []

    def protectedDiv(self, x, y):
        try:
            return x / y
        except ZeroDivisionError:
            return 1

    def protectedPow(self, x, y):
        try:
            return x**y
        except ZeroDivisionError:
            return 1

    def grade(self, expr):
        errs = 0
        func = self.toolbox.compile(expr=expr)
        try:
            for x, y in self.vals:
                if (func(x) != y):
                    errs += 1
            return errs / len(self.vals),
        except OverflowError:
            print("Overflowing on func!\n\t" + str(expr))
            return 1,
        except ZeroDivisionError:
            print("ZeroDivisionError on func!\n\t" + str(expr))
            return 1,

    def __init__(self):
        data = open(fp3,'r')
        data.readline()

        for line in data.readlines():
            x, y = line.split()
            self.vals.append((float(x), float(y)))

        domain = gp.PrimitiveSet('main', 1)
        domain.addPrimitive(operator.add, 2)
        # domain.addPrimitive(operator.sub, 2)
        domain.addPrimitive(operator.mul, 2)
        # domain.addPrimitive(self.protectedDiv, 2)
        # domain.addPrimitive(operator.pow, 2)
        # domain.addEphemeralConstant("rand", lambda: random.randint())
        domain.addTerminal(2)

        creator.create('FitnessMin', base.Fitness, weights=(-1.,))
        creator.create("Instance", gp.PrimitiveTree, fitness=creator.FitnessMin)

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


# Part 4


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
    
        # constants
    generation = 0
    max_generation = 40
    matingPB = .5
    mutatePB = .1

    # Init stats
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    # Init population
    train = Trainer4()
    toolbox = train.toolbox
    pop = toolbox.populate(n=300)
    hof = tools.HallOfFame(3)

    algorithms.eaSimple(pop, toolbox, matingPB, mutatePB, max_generation, stats=mstats, verbose=True, halloffame=hof)   
    for h in hof:
        print(h)


class Trainer4:
    toolbox = base.Toolbox()
    domain = gp.PrimitiveSet('main', 1)
    trainingSet = []
    testSet = []

    def protectedDiv(self, x, y):
        try:
            return x / y
        except ZeroDivisionError:
            return 1

    def protectedPow(self, x, y):
        try:
            return x**y
        except ZeroDivisionError:
            return 1

    def grade(self, expr):
        errs = 0
        func = self.toolbox.compile(expr=expr)
        for x, y in self.trainingSet:
            if (func(*x) != y):
                errs += 1
        return errs / len(self.trainingSet),

    def __init__(self):
        training = open(fp4[0],'r')
        test = open(fp4[1], 'r')
        
        for line in training.readlines():
            line = line.split()
            classification = line[-1]
            features = list(map(int,line[:-1]))
            self.trainingSet.append((features, classification))
        
        for line in test.readlines():
            line = line.split()
            classification = line[-1]
            features = list(map(int,line[:-1]))
            self.testSet.append((features, classification))
            

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
        toolbox.register("evaluate", self.grade)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mutate", gp.genFull, min_=1, max_=5)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mutate, pset=domain)
        toolbox.register("select", tools.selTournament, tournsize=3)
        # Prevents trees from getting too tall
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

        self.domain = domain
        self.toolbox = toolbox


if __name__ == '__main__':
    n = 3
    # n = input("which part do you want to run? ")
    if (int(n) == 1):
        perceptron()
    elif (int(n) == 2):
        packagedNeuralNetwork()
    elif (int(n) == 3):
        geneticFunction()
    elif (int(n) == 4):
        geneticClassification()
    else:
        perceptron()
        packagedNeuralNetwork()
        geneticFunction()
        geneticClassification()
    
    print("All functions run!")