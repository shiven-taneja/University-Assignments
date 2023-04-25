from q3.check_grad import check_grad
from q3.utils import *
from q3.logistic import *

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    #train_inputs, train_targets = load_train()
    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()

    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    hyperparameters = {
        "learning_rate": 0.1,
        "weight_regularization": 0.,
        "num_iterations": 265
    }
    weights = np.zeros((M+1, 1))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)
    # Begin learning with gradient descent
    #####################################################################
    # TODO:                                                             #
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################
    test_inputs, test_targets = load_test()
    CE_train = np.zeros((hyperparameters["num_iterations"], 1))
    correct_train = np.zeros((hyperparameters["num_iterations"], 1))
    CE_valid = np.zeros((hyperparameters["num_iterations"], 1))
    correct_valid = np.zeros((hyperparameters["num_iterations"], 1))
    CE_test = np.zeros((hyperparameters["num_iterations"], 1))
    correct_test = np.zeros((hyperparameters["num_iterations"], 1))
    for t in range(hyperparameters["num_iterations"]):
        f, df, y = logistic(weights, train_inputs, train_targets, hyperparameters)
        CE_train[t], correct_train[t] = evaluate(train_targets, y)
        weights -= (hyperparameters['learning_rate']/N)*df
        pred_valid = logistic_predict(weights, valid_inputs)
        CE_valid[t], correct_valid[t] = evaluate(valid_targets, pred_valid)
        pred_test = logistic_predict(weights, test_inputs)
        CE_test[t], correct_test[t] = evaluate(test_targets, pred_test)
    print(CE_train[-1], 1-correct_train[-1])
    print(CE_valid[-1], 1-correct_valid[-1])
    print(CE_test[-1], 1-correct_test[-1])
    plt.plot(range(0, hyperparameters['num_iterations']), CE_train, label='Training')
    plt.plot(range(0, hyperparameters['num_iterations']), CE_valid, label='Validation')
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Cross Entropy")
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    run_logistic_regression()

