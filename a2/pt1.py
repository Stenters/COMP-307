fp1 = 'ass2_data/part1/dataset'

def perceptron():
    """
    Part 1
        Report
            Report classification accuracy after 200 epochs
            Analyzse limitations for not achieving better results
    """
    # Vars
    epochs = 10
    net_err = 0
    weights = [0,0,0]
    learnRate = .1
    bias = 0
    data = []

    # Read data
    datafile = open(fp1, 'r')
    datafile.readline()

    # Parse data
    for line in datafile.readlines():
        data.append(((int(line[0]),int(line[2]),int(line[4])), int(line[6])))
    
    # Iterate through epochs
    while (epochs > 0):
        epochs -= 1

        # Initialize vars
        iterations = 0
        error = 0
        
        # Predict each line (online learning, so every prediction also changes the weights)
        for line in data:
            if predict(line[0], line[1], weights, bias, learnRate) != line[1]:
                error += 1
        
            iterations += 1
        
        error /= iterations
        net_err += error
        
    # When done, print results
    print(f"\n\tfinal bias: {bias}\n\tfinal weights: {weights}\n\tfinal error: {net_err / 10}")


def predict(data, classification, weights, bias, learnRate):
    # Sum up bias and weighted data
    sum = bias

    for i in range(0,len(data)):
        sum += data[i] * weights[i]

    # Output is either 0 or 1
    sum = 1 if sum >= 0 else 0

    # If incorrect, update bias and weights
    bias += learnRate * (classification - sum)

    for i in range(0, len(data)):
        weights[i] += learnRate * (classification - sum) * data[i]

    return sum


def run():
    perceptron()

if __name__ == '__main__':
    run()