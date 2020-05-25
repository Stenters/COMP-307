from sklearn.neural_network import MLPClassifier

fp2 = ('ass2_data/part2/wine', 'ass2_data/part2/wine_test', 'ass2_data/part2/wine_training')


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
    # Get data from file and init the classifier
    testFeatures, testClasses = parseFile(open(fp2[1], 'r')) 
    trainingFeatures, trainingClasses = parseFile(open(fp2[2], 'r'))
    classifier = MLPClassifier(solver='sgd', hidden_layer_sizes=(1,13), random_state=1, verbose=False)

    # Train the classifier, then predict the result
    classifier.fit(trainingFeatures, trainingClasses)
    res = classifier.predict(testFeatures)

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