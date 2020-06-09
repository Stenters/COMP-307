'''
TODO
    Initialize all counts to 1
    Take two file names as arguemnts
    sampleoutput.txt

    >>> pos
    [35.0, 31.0, 24.0, 32.0, 26.0, 19.0, 41.0, 40.0, 18.0, 35.0, 35.0, 41.0]
    >>> neg
    [54.0, 87.0, 52.0, 60.0, 51.0, 71.0, 76.0, 53.0, 37.0, 44.0, 88.0, 51.0]

'''
import sys


pos = []
neg = []
posClassWeight = 0
negClassWeight = 0


def trainNet(trainingSet):
    posFeatureCounts = [1] * len(trainingSet[0][0])
    negFeatureCounts = [1] * len(trainingSet[0][0])
    posClassCounts = 1
    negClassCounts = 1

    for i in trainingSet:
        if (i[1]):
            posFeatureCounts = [i[0][j] + posFeatureCounts[j] for j in range(len(posFeatureCounts))]
            posClassCounts += 1
        else:
            negFeatureCounts = [i[0][j] + negFeatureCounts[j] for j in range(len(negFeatureCounts))]
            negClassCounts += 1

    global pos
    global neg
    global posClassWeight
    global negClassWeight

    pos = [posFeatureCounts[i] / posClassCounts for i in range(len(posFeatureCounts))]
    posClassWeight = posClassCounts / len(trainingSet)
    
    neg = [negFeatureCounts[i] / negClassCounts for i in range(len(negFeatureCounts))]
    negClassWeight = negClassCounts / len(trainingSet)
    


def predict(features):
    '''
    P(class|features) = P(Class) * P(features|Class)
        P(Class) = classWeight (51/200)
        P(features|Class) = weights
    '''

    posVal = posClassWeight
    negVal = negClassWeight

    # val = 0
    for i in range(len(features)):
        if features[i]:
            posVal *= pos[i]
        else:
            negVal *= neg[i]

    return 0 if negVal > posVal else 1

def run(fp1, fp2):
    training = open(fp1, 'r')
    test = open(fp2, 'r')

    trainingSet = [(list(map(float,line.split()[:-1])), float(line.split()[-1])) for line in training.readlines()]
    trainNet(trainingSet)

    print("Positive weights: ", str(pos))
    print("Negative weights: ", str(neg))
    print("Positive class prob: ", str(posClassWeight))
    print("Negative class prob: ", str(negClassWeight))
    print()

    classes = [predict(list(map(float,i.split()))) for i in test.readlines()]

    print(classes)

if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2])
