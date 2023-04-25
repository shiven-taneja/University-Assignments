from q3.l2_distance import l2_distance
from q3.utils import *

import matplotlib.pyplot as plt
import numpy as np


def knn(k, train_data, train_labels, valid_data):
    """ Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples,
          M is the number of features per example.

    :param k: The number of neighbours to use for classification
    of a validation example.
    :param train_data: N_TRAIN x M array of training data.
    :param train_labels: N_TRAIN x 1 vector of training labels
    corresponding to the examples in train_data (must be binary).
    :param valid_data: N_VALID x M array of data to
    predict classes for validation data.
    :return: N_VALID x 1 vector of predicted labels for
    the validation data.
    """
    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:, :k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # Note this only works for binary labels:
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int)
    valid_labels = valid_labels.reshape(-1, 1)

    return valid_labels


def run_knn():
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    #####################################################################
    # TODO:                                                             #
    # Implement a function that runs kNN for different values of k,     #
    # plots the classification rate on the validation set, and etc.     #
    #####################################################################
    knn_1 = knn(1, train_inputs, train_targets, valid_inputs)
    knn_3 = knn(3, train_inputs, train_targets, valid_inputs)
    knn_5 = knn(5, train_inputs, train_targets, valid_inputs)
    knn_7 = knn(7, train_inputs, train_targets, valid_inputs)
    knn_9 = knn(9, train_inputs, train_targets, valid_inputs)
    knn_all = np.array([knn_1, knn_3, knn_5, knn_7, knn_9])

    classification_rates = np.zeros((5, 1))

    for k, c in zip(knn_all, classification_rates):
        for i, j in zip(k, valid_targets):
            if i == j:
                c += 1

    classification_rates = classification_rates / len(valid_targets)

    plt.bar([1, 3, 5, 7, 9], classification_rates.T[0])
    plt.xlabel('K')
    plt.ylabel('Classification Rate')
    plt.title("Classification Rate of K on Validation Set")
    plt.xticks([1, 3, 5, 7, 9])
    plt.show()

    ##FOR TEST DATA

    knn_test_3 = knn(3, train_inputs, train_targets, test_inputs)
    knn_test_5 = knn(5, train_inputs, train_targets, test_inputs)
    knn_test_7 = knn(7, train_inputs, train_targets, test_inputs)

    knn_test = np.array([knn_test_3, knn_test_5, knn_test_7])

    classification_rates_test = np.zeros((3, 1))

    for k_t, c_t in zip(knn_test, classification_rates_test):
        for i_t, j_t in zip(k_t, test_targets):
            if i_t == j_t:
                c_t += 1

    print(classification_rates_test/len(test_targets))



    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    run_knn()
