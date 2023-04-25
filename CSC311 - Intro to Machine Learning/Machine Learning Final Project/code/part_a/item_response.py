import numpy as np
import matplotlib.pyplot as plt
<<<<<<< HEAD
from starter_code.starter_code.utils import *
=======
from utils import *
>>>>>>> 3ca011e1b7f46fdf1af260d689c6a5ac959623c6


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    i = np.array(data['user_id'])
    j = np.array(data['question_id'])
    c = np.array(data['is_correct'])
    frac = sigmoid(theta[i]-beta[j])
    likelihood_array = np.log(frac)*c + np.log(1-frac)*(1-c)
    log_lklihood = np.sum(likelihood_array)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    i = np.array(data['user_id'])
    j = np.array(data['question_id'])
    c = np.array(data['is_correct'])
    new_theta = np.zeros(theta.shape)
    new_beta = np.zeros(beta.shape)
    for i_, j_, c_ in zip(i,j,c):
        new_theta[i_] += sigmoid(theta[i_]-beta[j_])-c_
        new_beta[j_] += c_-sigmoid(theta[i_]-beta[j_])
    theta -= new_theta*lr
    beta -= new_beta*lr
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(542)
    beta = np.zeros(1774)

    val_acc_lst = []
    log_like_train = []
    log_like_valid = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        neg_lld_val = neg_log_likelihood(val_data, theta=theta, beta=beta)
        log_like_train.append(-neg_lld)
        log_like_valid.append(-neg_lld_val)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, log_like_train, log_like_valid


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    theta, beta, val_acc_lst, log_like_train, log_like_valid = \
        irt(train_data, val_data, 0.015, 50)
    # plt.plot(log_like_train, label='Training Log-Likelihood')
    # plt.plot(log_like_valid, label='Validation Log-Likelihood')
    # plt.xlabel('Iterations')
    # plt.ylabel('Log-Likelihood')
    # plt.legend()
    # plt.show()
    #
    # val_acc = evaluate(val_data, theta, beta)
    # test_acc = evaluate(test_data, theta, beta)
    # print("Validation Accuracy: ", val_acc)
    # print("Test Accuracy:", test_acc)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    plt.scatter(theta, sigmoid(theta-beta[0]), label ='Question 1')
    plt.scatter(theta, sigmoid(theta-beta[3]), label ='Question 4')
    plt.scatter(theta, sigmoid(theta-beta[5]), label='Question 6')
    plt.xlabel('Theta')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
