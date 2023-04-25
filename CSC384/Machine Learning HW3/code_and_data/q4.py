'''
Question 4 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
import scipy.special as sp
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt


def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for i in range(0, 10):
        digits_by_label = data.get_digits_by_label(train_data, train_labels, i)
        mean_class = (np.sum(digits_by_label, axis=0))/digits_by_label.shape[0]
        means[i] += mean_class
    return means


def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    means = compute_mean_mles(train_data, train_labels)
    for i in range(0, 10):
        digits_by_label = data.get_digits_by_label(train_data, train_labels, i)
        x_mean = digits_by_label - means[i]
        cov_class = (np.dot(x_mean.T, x_mean))/len(digits_by_label)
        covariances[i] = cov_class + 0.01*np.eye(64)
    return covariances


def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    gen_like = np.zeros((digits.shape[0], 10))
    for i in range(0, 10):
        constant = np.log(2*np.pi)**(-(digits.shape[1])/2)
        log_sigma = np.log(np.linalg.det(covariances[i])**(-1/2))
        x_mean = digits - means[i]
        inverse = np.linalg.inv(covariances[i])
        third = np.log(np.exp(np.diag((-1/2)*x_mean.dot(inverse).dot(x_mean.T))))
        gen_like[:, i] = constant+log_sigma+third
    return gen_like


def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    gen_like = generative_likelihood(digits, means, covariances)
    y_k = np.log(1/10)
    log_numerator = gen_like+y_k
    log_denominator = sp.logsumexp(gen_like, axis=1, b=(1/10)).reshape(-1,1)
    con_like = log_numerator-log_denominator
    return con_like


def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute as described above and return
    log_avg = 0
    for i in range(0, len(digits)):
        log_avg += cond_likelihood[i][int(labels[i])]
    log_avg /= digits.shape[0]
    return log_avg


def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    return np.argmax(cond_likelihood, axis=1)


def accuracy(digits, labels, means, covariances):
    classify = classify_data(digits, means, covariances)
    acc = 0
    for i in range(0, len(classify)):
        if classify[i] == labels[i]:
            acc += 1
    acc /= len(classify)
    return acc


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation
    train_avg = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    test_avg = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    print("Average conditional log likelihood of training set:", train_avg)
    print("Average conditional log likelihood of test set:", test_avg)

    train_accuracy = accuracy(train_data, train_labels, means, covariances)
    test_accuracy = accuracy(test_data, test_labels, means, covariances)
    print("Test accuracy for the training set:", train_accuracy)
    print("Test accuracy for the test set:", test_accuracy)

    ev = []
    for i in range (0, 10):
        evalue, evector = np.linalg.eig(covariances[i])
        plt.imshow(evector[:, evalue.argmax()].reshape(8, 8), cmap='gray')
        plt.show()


if __name__ == '__main__':
    main()
