from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch
import matplotlib.pyplot as plt

def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        out = torch.sigmoid(self.g(inputs))
        out = torch.sigmoid(self.h(out))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function. 

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    # Create lists for storing data (NN)
    train_losses = []
    valid_accs = []

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.)

            loss += lamb / 2 * model.get_weight_norm()

            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))

        # Storage of accurate and loss data got after training
        train_losses.append(train_loss)
        valid_accs.append(valid_acc)
    return train_losses, valid_accs
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Question 3 c): Use latent dimensions of k ∈ {10, 50, 100, 200, 500}.

    # k_set = [10, 50, 100, 200, 500]
    # lr_set = [0.01, 0.03, 0.05, 0.1, 0.2]
    # for lr in lr_set:
    #     for k in k_set:
    #         print('K = ', k, 'lr = ', lr)
    #         model = AutoEncoder(train_matrix.shape[1], k)
    #         train(model, lr, 0, train_matrix, zero_train_matrix, valid_data, num_epoch=20)

    # We choose k = 10, learning rate = 0.1, the number of iterations = 10
    #
    #

    # Question 3 d): Use chosen k, plot and report how the training and validation objectives changes
    # Set model hyperparameters.
    k = 10
    model = AutoEncoder(train_matrix.shape[1], k)
    # Set optimization hyperparameters.
    lr = 0.1
    num_epoch = 10

    # # Start training and get the data of changes in training and validation objectives
    # train_losses, valid_accs = train(model, lr, 0, train_matrix, zero_train_matrix, valid_data, num_epoch)
    # f, ax = plt.subplots(1, 2)
    # ax[0].plot(range(num_epoch), train_losses)
    # ax[0].set_xlabel('Num of Epoch')
    # ax[0].set_ylabel('Train loss')
    #
    # ax[1].plot(range(num_epoch), valid_accs)
    # ax[1].set_xlabel('Num of Epoch')
    # ax[1].set_ylabel('Valid Accuracy')
    # plt.show()
    # # Report the final accuracy
    # acc = evaluate(model, zero_train_matrix, test_data)
    # print('Final test Accuracy is ', acc)

    # Question e): tune the regularization penalty in {0.001, 0.01, 0.1, 1}
    # train_losses, valid_accs = train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)

    # lamb_set = [0.001, 0.01, 0.1, 1]
    # for lamb in lamb_set:
    #     print('lamb = ', lamb)
    #     model = AutoEncoder(train_matrix.shape[1], k)
    #     train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)

    lamb = 0.001
    model = AutoEncoder(train_matrix.shape[1], k)
    train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)

    test_acc = evaluate(model, zero_train_matrix, test_data)
    print('Final test Accuracy is ', test_acc)
    valid_acc = evaluate(model, zero_train_matrix, valid_data)
    print('Final valid Accuracy is ', valid_acc)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

if __name__ == "__main__":
    main()
