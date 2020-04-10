import math
import sys

k = 1


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

    def __init__(self, Alcohol, Malic_acid, Ash, Alcalinity_of_ash, Magnesium, Total_phenols, Flavanoids,
         Nonflavanoid_phenols, Proanthocyanins, Color_intensity, Hue, OD280_OD315_of_diluted_wines, Proline, Class=None):
        self.Alcohol = Alcohol
        self.Malic_acid = Malic_acid
        self.Ash = Ash
        self.Alcalinity_of_ash = Alcalinity_of_ash
        self.Magnesium = Magnesium
        self.Total_phenols = Total_phenols
        self.Flavanoids = Flavanoids
        self.Nonflavanoid_phenols = Nonflavanoid_phenols
        self.Proanthocyanins = Proanthocyanins
        self.Color_intensity = Color_intensity
        self.Hue = Hue
        self.OD280_OD315_of_diluted_wines = OD280_OD315_of_diluted_wines
        self.Proline = Proline

    def getDist(self, otherData):
        return math.sqrt(
            (((self.Alcohol - otherData.Alcohol)**2) / (ranges[0]**2)) +
            (((self.Malic_acid - otherData.Malic_acid)**2) / (range[1]**2)) +
            (((self.Ash - otherData.Ash)**2) / (range[2]**2)) +
            (((self.Alcalinity_of_ash - otherData.Alcalinity_of_ash)**2) / (range[3]**2)) +
            (((self.Magnesium - otherData.Magnesium)**2) / (range[4]**2)) +
            (((self.Total_phenols - otherData.Total_phenols)**2) / (range[5]**2)) +
            (((self.Flavanoids - otherData.Flavanoids)**2) / (range[6]**2)) +
            (((self.Nonflavanoid_phenols - otherData.Nonflavanoid_phenols)**2) / (range[7]**2)) +
            (((self.Proanthocyanins - otherData.Proanthocyanins)**2) / (range[8]**2)) +
            (((self.Color_intensity - otherData.Color_intensity)**2) / (range[9]**2)) +
            (((self.Hue - otherData.Hue)**2) / (range[10]**2)) +
            (((self.OD280_OD315_of_diluted_wines - otherData.OD280_OD315_of_diluted_wines)**2) / (range[11]**2)) +
            (((self.Proline - otherData.Proline)**2) / (range[12]**2))
        )



def main():
    if len(sys.argv) != 3:
        print("Wrong number of arguments! please input a training file and a test file")
    
    else:
        (training, test) = sys.argv[1], sys.argv[2]

        training = open(training, 'r')
        test = open(test, 'r')

        dataSet = parseFile(training)
        test = parseFile(test, dataSet)
        
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
            float(vals[0]),
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
            float(vals[12]),
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

    res = []
    lines = file.readlines()

    for line in lines[1:]:
        vals = line.rstrip().split(" ")
        d = Data(
            float(vals[0]),
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
        )

        mindist = [9999] * k
        for data in dataSet:
            tmpDist = d.getDist(data)
            for dist in mindist:
                if tmpDist < dist:
                    dist = tmpDist


        res.append(d)

    return res

if __name__ == "__main__":
    main()