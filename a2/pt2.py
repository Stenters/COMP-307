from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

fp2 = ('ass2_data/part2/wine_test', 'ass2_data/part2/wine_training')


def packagedNeuralNetwork():
    # Get data from file and init the classifier
    testFeatures, testClasses = parseFile(open(fp2[0], 'r')) 
    trainingFeatures, trainingClasses = parseFile(open(fp2[1], 'r'))
    classifier = MLPClassifier(solver='sgd', max_iter=1000)

    # Preprocessing
    scaler = StandardScaler()
    scaler.fit(trainingFeatures)
    trainingFeatures = scaler.transform(trainingFeatures)
    testFeatures = scaler.transform(testFeatures)

    # Train the classifier, then predict the result
    classifier.fit(trainingFeatures, trainingClasses)
    res = classifier.predict(testFeatures)
    accuracy = accuracy_score(testClasses, res)
    print("accuracy of prediction: ", accuracy)


def parseFile(file):
    # split the line on spaces, cast to float, last val is the class, others are features
    features = []
    classes = []
    file.readline()
    for line in file.readlines():
        vals = list(map(float, line.split()))
        features.append(vals[:-1])
        classes.append(vals[-1])
    return features, classes


def run():
    packagedNeuralNetwork()
    
if __name__ == '__main__':
    run()