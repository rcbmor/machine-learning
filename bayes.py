# Example of Naive Bayes implemented from Scratch in Python

import sys
import csv
import random
import math


def load_csv(filename):
    """
    loads csv to a data_set
    :param filename: csv file input

    :return data_set: a list of items converted to float
        data_set = [
            [0,1,2,3,1,
            [0,1,2,3,0],
            [0,1,2,4,1],
            ...]
    """
    lines = csv.reader(open(filename, "rb"))
    data_set = list(lines)
    for i in range(len(data_set)):
        data_set[i] = [float(x) for x in data_set[i]]
    return data_set


def split_data_set(data_set, split_ratio):
    """
    splits the data_set to create the training and test sets
    :param data_set: the loaded data_set
    :param split_ratio: how much on each set
    :return [train_set, test_set]: the training and test sets
    """
    train_size = int(len(data_set) * split_ratio)
    train_set = []
    copy = list(data_set)
    while len(train_set) < train_size:
        index = random.randrange(len(copy))
        train_set.append(copy.pop(index))
    return [train_set, copy]


def separate_by_class(data_set):
    """
    separate all items into classes
    :param data_set: input data set
        data_set = [
            [0,1,2,3,1],
            [0,1,2,3,0],
            [0,1,2,4,1],
            ...]
    :return: hash of classes with respective vector occurrences
        {
        0: [[0, 1, 2, 3, 0]],
        1: [[0, 1, 2, 3, 1], [0, 1, 2, 4, 1]]
        }
    """
    separated = {}
    for i in range(len(data_set)):
        vector = data_set[i]
        # vector pos -1 is the class
        if vector[-1] not in separated:
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


def mean(numbers):
    """
    calculate the mean of a list of numbers
    :param numbers: list of numbers
        [1, 2, 3]
    :return: float mean
        2.0
    """
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    """
    calculate the stdev of a list of numbers
    :param numbers: list of numbers
        [10, 5, 1, 2, 3]
    :return: float stdev
        3.5637059362410923
    """
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


def summarize(data_set):
    """
    calculate mean and stdev for each i-th item from data_set
    how zip works:
        zipped = [[1,2],[3,4],[5,6]]
        zip(*zipped)
        [(1, 3, 5), (2, 4, 6)]
    :param data_set: input data set
        data_set = [
            [0,1,2,3,'a'],
            [0,1,2,3,'b'],
            [0,1,2,4,'c'],
            ...]
    :return:
    """
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*data_set)]
    del summaries[-1]
    return summaries


def summarizeByClass(dataset):
    separated = separate_by_class(dataset)
    summaries = {}
    for classValue, instances in separated.iteritems():
        summaries[classValue] = summarize(instances)
    return summaries


def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities


def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.iteritems():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def main():
    filename = sys.argv[1]
    splitRatio = 0.67
    dataset = load_csv(filename)
    trainingSet, testSet = split_data_set(dataset, splitRatio)
    print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
    # prepare model
    summaries = summarizeByClass(trainingSet)
    # test model
    predictions = getPredictions(summaries, testSet)
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: {0}%').format(accuracy)


main()