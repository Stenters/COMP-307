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
import sklearn
import deap

# File Paths
fp1 = 'ass2_data/part1/dataset'
fp2 = ('ass2_data/part2/wine', 'ass2_data/part2/wine_test', 'ass2_data/part2/wine_training')
fp3 = 'ass2_data/part3/regression'
fp4tmp = 'ass2_data/part4/satellite'
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
    data = open(fp1, 'r')
    perceptron = Perceptron(('f1','f2','f3'),('c1','c2'))
    print(perceptron)


class Perceptron:

    def __init__(self, features, classes):
        self.output = self.Node(0)
        self.features = []
        self.classes = classes

        for i in range(1, len(features) + 1):
            x = self.Node(i)
            self.output.weights.append((x,0))


    def classify(self, featureVector):
        pass

    def __repr__(self):
        return str(self.output)


    class Node:
        increment = .2

        def __init__(self, index):
            self.index = index
            self.weights = []
            self.bias = 0

        def updateWeights(self, features, classification):
            prediction = self.predict(features)
            self.bias += self.increment * (classification - prediction)
            
            for i in range(0, len(self.weights)):
                self.weights[i][1] += self.increment * (classification - prediction) * self.weights[i][0].predict(features)

        def predict(self, features):
            sum = self.bias
            for i in range(0, len(self.weights)):
                sum += self.weights[i][0].predict(features[i]) * self.weights[i][1]
            
            return [1 if sum > 0 else 0]

        def __repr__(self):
            return f"{self.index}: {[ str(x[0].index) + ', ' + str(x[1]) for x in self.weights]}"


### Part 2


def packagedNeuralNetwork():
    """
    Part 2
        Train a neural network to classify data using an existing package
            Keras?
        Reformat data for package
        Report
            Determine the network architecture (# inoput nodes, # output, # hidden nodes (assume one layer)). Describe rationale
            Determine learing parameters (inc learning rate, momentum, initial weight ranges, etc). Describe rationale
            Determine termination criteria. Describe rationale
            Report reults (10 independent experiments with different random seeds) on training and test. Analyse results and make conclusions
            Compare vs nearest neighbor
            Describe why you used that package
    """
    pass


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
    pass


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
    pass


if __name__ == '__main__':
    perceptron()
    packagedNeuralNetwork()
    geneticFunction()
    geneticClassification()
    print("All functions run!")