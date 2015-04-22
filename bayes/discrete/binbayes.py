__author__ = 'ricardo.moreira'

import csv
import random
import argparse
from collections import Counter, defaultdict

debug = False


def split_data_set(data_set, split_ratio=0.67):
    """
    splits the data_set to create the training and test sets
    :param data_set: the loaded data_set
    :param split_ratio: how much on each set (default 0.67)
    :return [train_set, test_set]: the training and test sets
    """
    train_size = int(len(data_set) * split_ratio)
    train_set = []
    test_set = list(data_set)
    while len(train_set) < train_size:
        index = random.randrange(len(test_set))
        train_set.append(test_set.pop(index))
    return [train_set, test_set]


def split_data_set_infolds(data_set, nfolds, randomize=True):
    """

    :param data_set:
    :param nfolds:
    :return: testing_this_round, testing_this_round
    """
    num_folds = int(nfolds)
    subset_size = len(data_set) / num_folds
    if randomize:
        from random import shuffle

        data_set = list(data_set)
        shuffle(data_set)
    for n in range(num_folds):
        i = n * subset_size
        j = (n + 1) * subset_size
        testing_this_round = data_set[i:][:subset_size]
        training_this_round = data_set[:i] + data_set[j:]
        yield testing_this_round, testing_this_round


def load_csv(filename):
    """
    loads csv to a data_set
    :param filename: csv file input
        attr0,attr1,attr2,attr3,class1
        attr0,attr1,attr2,attr3,class1
        attr0,attr1,attr2,attr3,class1
        ...

    :return  data_set: a list of instances as array of attributes
        data_set = [
            ['attr0','attr1','attr2','attr3','class1'],
            ['attr0','attr1','attr2','attr3','class2'],
            ['attr0','attr1','attr2','attr3','class1'],
            ...
        ]
    """
    if isinstance(filename, file):
        lines = csv.reader(filename)
    else:
        lines = csv.reader(open(filename, "rb"))
    data_set = list(lines)
    for i in range(1, len(data_set)):
        data_set[i] = [x for x in data_set[i]]
    return data_set


def load_data_set_kfolds(filename, nfolds=10):
    """
    load a data set with holdout strategy
    :param filename: the csv file to be loaded
    :return: (train_data_set, test_data_set)
    """
    return split_data_set_infolds(load_csv(filename), nfolds)


def load_data_set_file(filename):
    """
    load a data set with holdout strategy
    :param filename: the csv file to be loaded
    :return: (train_data_set, test_data_set)
    """
    return split_data_set(load_csv(filename))


def train_classifier(labels, train_data_set):
    """
    calculate prior probabilities and likelihood for each attribute

    :param: labels: a list of attribute labels (names)
        [ 'name0','name1','name2','name3','classname']

    :param: train_data_set: a list of instances as array of attributes
        data_set = [
            ['attr0','attr1','attr2','attr3','class1'],
            ['attr0','attr1','attr2','attr3','class2'],
            ['attr0','attr1','attr2','attr3','class1'],
            ...
        ]

    :return:
        priors: Counter, count each class frequency
            Counter( {
                'class1': 2,
                'class2': 1
                 })
        likelihood: dictionary of dictionary, for each attribute the frequency of values
            {
             'name0': {'attr0': 3, 'attr0x': 6}),
             'name1': {'attr1': 2, 'attr1x': 4, 'attr1y': 3}
             'name1': {'attr2': 2, 'attr2y': 3}
            }
    """
    priors = Counter()
    likelihood = {}

    for item in train_data_set:
        klass = item[-1]
        priors[klass] += 1  # last is the class
        likelihood.setdefault(klass, {})

        for i in range(len(labels) - 1):
            label = labels[i]
            attr = item[i]
            likelihood[klass].setdefault(label, defaultdict(int))
            likelihood[klass][label][attr] += 1

    return priors, likelihood


def classify_max_prior(item, priors, likelihood):
    """Basic classifier for testing, return the max class"""
    return max(priors, key=lambda x: priors[x])


def classify_bayesian(item, priors, likelihood, labels):
    """
    Naive Bayes
    Returns the class that maximizes the posterior probability

    :param: item to be classified
        ['attr0','attr1','attr2','attr3','class1'],

    :param: priors
        priors: Counter, count each class frequency
            Counter( {
                'class1': 2,
                'class2': 1
                 })
    :param: likelihood tables
        likelihood: dictionary of dictionary, for each attribute the frequency of values
            {
             'name0': {'attr0': 3, 'attr0x': 6}),
             'name1': {'attr1': 2, 'attr1x': 4, 'attr1y': 3}
             'name1': {'attr2': 2, 'attr2y': 3}
            }
    :param: labels: a list of attribute labels (names)
        [ 'name0','name1','name2','name3','classname']

    :return: the predicted class
    """
    max_class = (1E-6, '')  # 0.000001
    t = sum(priors.values())
    for c in priors.keys():
        tc = priors[c]
        p = float(tc) / t  # P(C)
        if debug: print "Likelihood:", c, ":", tc, "/", t, "=", p
        for i in range(len(labels) - 1):
            ta = likelihood[c][labels[i]][item[i]]
            pa = float(ta) / tc  # P(A_i|C)
            p = p * pa
            if debug: print "Likelihood:", c, labels[i], item[i], ":", ta, "/", tc, "=", pa

        if debug: print "Likelihood result:", c, "=", p

        if p > max_class[0]:
            max_class = (p, c)

    return max_class[1]


def accuracy(truth, predicted):
    """
    how many we predicted right?
    :param truth: list of right values
        [yes, yes, no, yes]
    :param predicted: list of predicted
        [yes, no, no, no]
    :return: float number of hits / total
        0.5
    """
    if len(truth) != len(predicted):
        raise Exception("Wrong sizes")

    total = len(truth)
    if total == 0:
        return 0

    hits = len(filter(lambda (x, y): x == y, zip(truth, predicted)))
    return float(hits) / total


def calculate_performance(confusion_matrix):
    """

    :param confusion_matrix:
        matrix['T'] = (tp, tn)
        matrix['F'] = (fp, fn)

    :return:
    """
    performance = {}
    tp = confusion_matrix['T'][0]
    tn = confusion_matrix['T'][1]
    fp = confusion_matrix['F'][0]
    fn = confusion_matrix['F'][1]
    if debug: print "TP:", tp, "TN:", tn, "FP", fp, "FN", fn

    performance['Precision'] = float(tp) / (tp + tn)
    performance['Accuracy'] = float(tp + tn) / (tp + tn + fp + fn)
    performance['Revocation'] = float(tp) / (tp + fn)
    performance['F-Score'] = float(2) / ((1 / performance['Accuracy']) + (1 / performance['Revocation']))
    return performance


def calculate_confusion_matrix(truth, predicted, positive):
    """
    harmonic mean of precison and sensitivity
    :param truth: list of right values
        [yes, yes, no, yes]
    :param predicted: list of predicted
        [yes, no, no, no]
    :param positive: value to consider as positive
        "yes"
    :return:  confusion_matrix:
        matrix['T'] = (tp, tn)
        matrix['F'] = (fp, fn)

    """
    if len(truth) != len(predicted):
        raise Exception("Wrong sizes")

    total = len(truth)
    if total == 0:
        return 0

    tp = len(filter(lambda (x, y): x == y and x == positive, zip(truth, predicted)))
    tn = len(filter(lambda (x, y): x == y and x != positive, zip(truth, predicted)))

    fp = len(filter(lambda (x, y): x != y and x != positive, zip(truth, predicted)))
    fn = len(filter(lambda (x, y): x != y and x == positive, zip(truth, predicted)))

    matrix = {}
    matrix['T'] = (tp, tn)
    matrix['F'] = (fp, fn)

    return matrix


def run_classifier(test_data_set, priors, likelihood, labels, positive):
    """
    classfify each item in test data set and calculate all required metrics
    :param: test data set
            data_set = [
            ['attr0','attr1','attr2','attr3','class1'],
            ['attr0','attr1','attr2','attr3','class2'],
            ['attr0','attr1','attr2','attr3','class1'],
            ...
        ]

    :param: what analysis we should do? holdout or kfolds
    :param: prior class frequency
        priors: Counter, count each class frequency
            Counter( {
                'class1': 2,
                'class2': 1
                 })
    :param: likelihood attribute frequency
        likelihood: dictionary of Counters, for each attribute frequency
            { 'class2': Counter( {'attr2': 1, 'attr3': 1, 'attr0': 1, 'attr1': 1}),
              'class1': Counter( {'attr2': 2, 'attr3': 2, 'attr0': 2, 'attr1': 2}),
              ...
            }

    :return: confusion_matrix

    """

    predicted = []
    truth = []
    for item in test_data_set:
        t = item[-1]
        p = classify_bayesian(item[:-1], priors, likelihood, labels)
        # p = classify_max_prior(item[:-1], priors, likelihood)
        truth.append(t)
        predicted.append(p)
        if debug: print "Item: ", item, " class: ", t, " predicted: ", p

    if debug: print "Ruf accuracy:", accuracy(truth, predicted)

    confusion_matrix = calculate_confusion_matrix(truth, predicted, positive)
    # performance = calculate_performance(confusion_matrix)

    return confusion_matrix


def print_result_confusion_matrix(confusion_matrix):
    print "Confusion Matrix"
    print " ", confusion_matrix.keys()
    for k, v in confusion_matrix.iteritems():
        print "{:<3}{:<5}".format(k, v)
    print


def print_result_performance(performance):
    print "Performance: "
    for k, v in performance.iteritems():
        print "{:<12}= {:<5.4}".format(k, v)


def add_results(confusion_matrix, k_confusion_matrix):
    """

    :param confusion_matrix:
    :param k_confusion_matrix:
    :return:
    """
    confusion_matrix['T'] = (confusion_matrix['T'][0] + k_confusion_matrix['T'][0],
                             confusion_matrix['T'][1] + k_confusion_matrix['T'][1])
    confusion_matrix['F'] = (confusion_matrix['F'][0] + k_confusion_matrix['F'][0],
                             confusion_matrix['F'][1] + k_confusion_matrix['F'][1])


def main():
    description = "Naive Beyes Classifier - Ricardo Moreira and Ian Rosadas"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--analysis', help='chose performance analysis', choices=['holdout', 'kfolds'])
    parser.add_argument('--dataset', type=argparse.FileType('rt'), help='the csv file of the dataset')
    parser.add_argument('--names', type=argparse.FileType('rt'), help='a csv file with attribute names')
    parser.add_argument('--positive', help='value for positive class')
    args = parser.parse_args()

    if args.test:
        data_set_file = "./datasets/tic-tac-toe.data"
        names_file = "./datasets/tic-tac-toe.names"
        analysis = "kfolds"
        # analysis = "holdout"
        positive = "positive"
        data_set_file = "./datasets/playgolf.data"
        names_file = "./datasets/playgolf.names"
        analysis = "holdout"
        positive = "yes"
    elif not args.dataset or not args.names or not args.positive or not args.analysis:
        parser.print_help()
        exit(-1)
    else:
        data_set_file = args.dataset
        names_file = args.names
        analysis = args.analysis
        positive = args.positive

    performance = {'F-Score': 0.0, 'Precision': 0.0, 'Revocation': 0.0, 'Accuracy': 0.0}
    confusion_matrix = {'T': (0, 0), 'F': (0, 0)}

    labels = load_csv(names_file)[0]
    if debug: print "Labels ", labels

    if analysis == "holdout":
        train_data_set, test_data_set = load_data_set_file(data_set_file)
        if debug: print "Train data set", len(train_data_set), train_data_set[:3], "..."
        if debug: print "Test data set", len(test_data_set), test_data_set[:3], "..."

        priors, likelihood = train_classifier(labels, train_data_set)
        if debug: print "Priors ", priors
        if debug: print "Likelihood ", likelihood

        confusion_matrix = run_classifier(test_data_set, priors, likelihood, labels, positive)
        print "Dataset: train =", len(train_data_set), ", test =", len(test_data_set)
        print

    elif analysis == "kfolds":
        i = 1

        for train_data_set, test_data_set in load_data_set_kfolds(data_set_file, 3):

            print "Dataset:", i, ", train =", len(train_data_set), ", test =", len(test_data_set)
            priors, likelihood = train_classifier(labels, train_data_set)
            if debug: print "Priors ", priors
            if debug: print "Likelihood ", likelihood

            k_confusion_matrix = run_classifier(test_data_set, priors, likelihood, labels, positive)
            if debug: print_result_confusion_matrix(k_confusion_matrix)

            add_results(confusion_matrix, k_confusion_matrix)
            i += 1

    performance = calculate_performance(confusion_matrix)
    print_result_confusion_matrix(confusion_matrix)
    print_result_performance(performance)


if __name__ == '__main__':
    main()