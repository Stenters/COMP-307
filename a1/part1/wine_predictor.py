import math
import sys


class Data:
    ranges = [
        3.8,
        5.06,
        1.87,
        19.4,
        92,
        2.9,
        4.74,
        .53,
        3.17,
        11.72,
        1.23,
        2.73
    ]

    vals = []
    Class = None
    k = 7


    def __init__(self, vals, Class=None):
        self.vals = vals
        self.Class = Class

    def __repr__(self):
        return f"{self.vals} ({self.Class}) [{[x[1].Class for x in self.neighbors]}]"

    def getDist(self, otherData):
        res = 0
        for i in range(len(self.vals)-1):
            res += ((self.vals[i] - otherData.vals[i])**2 / (self.ranges[i])**2)

        return math.sqrt(res)

    def clasify(self, dataSet):
        potentials = []
        for dataPoint in dataSet:
            dist = self.getDist(dataPoint)
            potentials.append([dist, dataPoint])
        potentials.sort()
        self.neighbors = potentials[:self.k] 
        neighborClasses = list(map(lambda x: x[1].Class, self.neighbors))
        self.Class = max(set(neighborClasses), key=neighborClasses.count)


def main():
    if len(sys.argv) != 3:
        print("Wrong number of arguments! please input a training file and a test file")
    
    else:
        (training, test) = sys.argv[1], sys.argv[2]

        training = open(training, 'r')
        test = open(test, 'r')

        dataSet = parseTrainingFile(training)
        print(len(dataSet))
        results = parseTestFile(test, dataSet)

        print([r[0].Class for r in results])

        grade(results)


def parseTrainingFile(file):
    """
    Method for parsing a classified data file
    First line: class names
    Subsequent lines: data        
    """

    res = []
    lines = file.readlines()

    for line in lines[1:]:
        vals = line.rstrip().split(" ")
        d = Data(
            [float(vals[0]),
            float(vals[1]),
            float(vals[2]),
            float(vals[3]),
            float(vals[4]),
            float(vals[5]),
            float(vals[6]),
            float(vals[7]),
            float(vals[8]),
            float(vals[9]),
            float(vals[10]),
            float(vals[11]),
            float(vals[12])
            ],
            float(vals[13])
        )
        res.append(d)

    return res


def parseTestFile(file, dataSet):
    """
    Method for parsing an unclassified data file
    First line: class names
    Subsequent lines: data        
    """

    lines = file.readlines()
    testData = []

    for line in lines[1:]:
        vals = line.rstrip().split(" ")
        d = Data(
            [float(vals[0]),
            float(vals[1]),
            float(vals[2]),
            float(vals[3]),
            float(vals[4]),
            float(vals[5]),
            float(vals[6]),
            float(vals[7]),
            float(vals[8]),
            float(vals[9]),
            float(vals[10]),
            float(vals[11]),
            float(vals[12])]
        )

        d.clasify(dataSet)
        testData.append([d, float(vals[13])])

    return testData


def grade(results):
    errors = 0

    for r in results:
        if r[0].Class != r[1]:
            print(f"wrong classification: {r[1]} was {r[0]}")
            errors += 1

    print(f"errors: {errors}\ntotal: 89\n%accuracy: {(1-(errors/89))*100}")


if __name__ == "__main__":
    main()