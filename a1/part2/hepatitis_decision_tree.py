import math
import sys

# Constants

mostProbableNode = None

# Classes

class Node:
    def __init__(self, attribute, left, right):
        self.attribute = attribute
        self.left = left
        self.right = right

    def __repr__(self):
        l, r = '', ''
        if type(self.left) == Leaf:
            l = self.left.Class
        elif type(self.left) == Node:
            l = self.left.attribute

        if type(self.right) == Leaf:
            r = self.right.Class
        elif type(self.right) == Node:
            r = self.right.attribute
        
        return f'[{self.attribute}] ({l},{r})'


class Leaf:
    def __init__(self, Class, probablility):
        self.Class = Class
        self.probablility = probablility


class Instance:
    def __init__(self, attributes, Class=None):
        self.attributes = attributes
        self.Class = Class

    def __repr__(self):
        return f"{[self.attributes[a] for a in self.attributes]}, {self.Class}"

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

        printTree(decisionTree)


def BuildTree(instances, attributes):
    if len(instances) == 0:
        return mostProbableNode

    if len(set(map(lambda x: x.Class, instances))) == 1:
        return makeLeafNode(instances)

    if len(attributes) == 0:
        return makeLeafNode(instances)

    else:
        bestPurity = 1

        for a in attributes:
            yes, no = [],[]

            for i in instances:
                if i.attributes[a]:
                    yes.append(i)
                else:
                    no.append(i)

            yespurity = getImpurity(yes) * (len(yes) / len(instances))
            nopurity = getImpurity(no) * (len(no) / len(instances))
            min_weight_avg_impurity = yespurity + nopurity

            if min_weight_avg_impurity < bestPurity:
                bestPurity = min_weight_avg_impurity
                bestAtt = a
                bestInstTrue = yes
                bestInstFalse = no 

        attributes.remove(bestAtt)
        left = BuildTree(bestInstTrue, attributes)
        right = BuildTree(bestInstFalse, attributes)

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

    for line in lines[1:]:
        vals = line.rstrip().split(" ")
        
        for i in range(1,len(vals)):
            if vals[i] == 'true':
                vals[i] = True
            else:
                vals[i] = False

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

        res.append(d)
    
    global mostProbableNode
    mostProbableNode = makeLeafNode(res)

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

        for i in range(1,len(vals)):
            if vals[i] == 'true':
                vals[i] = True
            else:
                vals[i] = False

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
            errors += 1

    print(f"errors: {errors}\ntotal: {len(results)}\n%accuracy: {(1-(errors/len(results)))*100}")

# Helper Methods

def makeLeafNode(instances):
    classes = list(map(lambda x: x.Class, instances))
    mostCommonClass = max(set(classes), key=classes.count)
    prob = classes.count(mostCommonClass) / len(classes)
    return Leaf(mostCommonClass, prob)


def classify(instance, tree):
    # print(f'tree is {tree}')
    if type(tree) == Leaf:
        instance.Class = tree.Class
        return instance
    elif instance.attributes[tree.attribute]:
        return classify(instance, tree.left)
    else:
        return classify(instance, tree.right)


def getImpurity(instanceList):
    if len(instanceList) == 0:
        return 0

    numIsPresent, numIsNotPresent = 0,0

    for i in instanceList:
        if i.Class == 'live':
            numIsPresent += 1
        else:
            numIsNotPresent += 1
    
    return (numIsPresent * numIsNotPresent) / (numIsPresent + numIsNotPresent)**2


def getBestPurity(yesPurity, noPurity, prevPurity):
    return yesPurity #(FIXME)


def printTree(root):
    printBranch(root, '')


def printBranch(node, indent):
    if type(node) == Leaf:
        print(f"{indent}Class {node.Class}, prob = {node.probablility:.2f}")
    else:
        print(f"{indent}{node.attribute} = True:")
        printBranch(node.left, indent + '\t')
        print(f"{indent}{node.attribute} = False:")
        printBranch(node.right, indent + '\t')        


if __name__ == "__main__":
    main()