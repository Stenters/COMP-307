import math
import sys

# Constants

mostProbableNode = None
bestPurity = True

# Classes

class Node:
    def __init__(self, attribute, left, right, Class=None):
        self.attribute = attribute
        self.left = left
        self.right = right
        self.Class = Class

    def isLeaf(self):
        return self.left == None and self.right == None


class Instance:
    def __init__(self, attributes, Class=None):
        self.attributes = attributes
        self.Class = Class

# Methods

def main():
    if len(sys.argv) != 3:
        print("Wrong number of arguments! please input a training file and a test file")
    
    else:
        (training, test) = sys.argv[1], sys.argv[2]

        training = open(training, 'r')
        test = open(test, 'r')

        decisionTree = parseTrainingFile(training)
        results = parseTestFile(test, decisionTree)
        grade(results)


def BuildTree(instances, attributes):
    if len(instances) == 0:
        return mostProbableNode

    if len(set(map(lambda x: x.Class, instances))) == 1:
        return instances[0].Class

    if len(attributes) == 0:
        instanceClasses = list(map(lambda x: x.Class, instances))
        return max(set(instanceClasses), key=instanceClasses.count)

    else:
        for a in attributes:
            yes, no = [],[]

            for i in instances:
                if i.attributes[a]:
                    yes.append(i)
                else:
                    no.append(i)

            yespurity = getPurity(yes, a, True)
            nopurity = getPurity(no, a, False)

            netPurity = getBestPurity(yespurity, nopurity, bestPurity)
            if netPurity > bestPurity:
                bestPurity = netPurity
                bestAtt = a
                bestInstTrue = yes
                bestInstFalse = no 

        left = BuildTree(bestInstTrue, attributes.remove(bestAtt))
        right = BuildTree(bestInstFalse, attributes.remove(bestAtt))

    return Node(bestAtt, left, right)


def parseTrainingFile(file):
    """
    Method for parsing a classified data file
    First line: attribute names
    Subsequent lines: data
    Class AGE FEMALE STEROID ANTIVIRALS FATIGUE MALAISE ANOREXIA BIGLIVER FIRMLIVER SPLEENPALPABLE SPIDERS ASCITES VARICES BILIRUBIN SGOT HISTOLOGY
    """

    res = []
    lines = file.readlines()
    classes = []

    for line in lines[1:]:
        vals = line.rstrip().split(" ")
        d = Instance({
            'age': vals[1],
            'female': vals[2],
            'steroid': vals[3],
            'antivirals': vals[4],
            'fatigue': vals[5],
            'malaise': vals[6],
            'anorexia': vals[7],
            'bigliver': vals[8],
            'firmliver': vals[9],
            'spleenpalpable': vals[10],
            'spiders': vals[11],
            'ascites': vals[12],
            'varices': vals[13],
            'bilirubin': vals[14],
            'sgot': vals[15],
            'histology': vals[16]
        }, vals[0])

        classes.append(vals[0])
        res.append(d)

    mostProbableNode = Node(None,None,None,max(set(classes), key=classes.count))

    return BuildTree(res, [
        'age', 'female', 'steroid', 'antivirals', 'fatigue', 'malaise', 
        'anorexia', 'bigliver', 'firmliver', 'spleenpalpable', 'spiders', 
        'ascites', 'varices', 'bilirubin', 'sgot', 'histology'
    ])


def parseTestFile(file, decisionTree):
    """
    Method for parsing an unclassified data file
    First line: class names
    Subsequent lines: data        
    """

    lines = file.readlines()
    testData = []

    for line in lines[1:]:
        vals = line.rstrip().split(" ")
        d = Instance({
            'age': vals[1],
            'female': vals[2],
            'steroid': vals[3],
            'antivirals': vals[4],
            'fatigue': vals[5],
            'malaise': vals[6],
            'anorexia': vals[7],
            'bigliver': vals[8],
            'firmliver': vals[9],
            'spleenpalpable': vals[10],
            'spiders': vals[11],
            'ascites': vals[12],
            'varices': vals[13],
            'bilirubin': vals[14],
            'sgot': vals[15],
            'histology': vals[16]
        })
        testData.append([classify(d,decisionTree), vals[0]])

    return testData


def grade(results):
    errors = 0

    for r in results:
        if r[0].Class != r[1]:
            print(f"wrong classification: {r[1]} was {r[0]}")
            errors += 1

    print(f"errors: {errors}\ntotal: 25\n%accuracy: {(1-(errors/25))*100}")

# Helper Methods

def classify(instance, tree):

    if tree.Class != None:
        instance.Class = tree.Class
    elif instance.attributes[tree]:
        classify(instance, tree.left)
    else:
        classify(instance, tree.right)


def getPurity(instanceList, attribute, isPresent):
    return 0


def getBestPurity(yesPurity, noPurity, prevPurity):
    return 0

if __name__ == "__main__":
    main()