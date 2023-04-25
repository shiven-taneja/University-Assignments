# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as special
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist



#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    ## TODO
    a_nume = np.exp(-1*np.divide(l2(np.reshape(test_datum, (1, test_datum.shape[0])), x_train), 2*tau**2))
    a_deno = np.exp(special.logsumexp(-1*np.divide(l2(np.reshape(test_datum, (1, test_datum.shape[0])), x_train), 2*tau**2)))
    A = np.divide(a_nume, a_deno)
    A = np.diag(A[0])
    aa = np.dot(np.dot(x_train.T, A), x_train)
    bb = np.dot(np.dot(x_train.T, A), y_train)
    w = np.linalg.solve(aa + lam*np.identity(len(x_train[0])), bb)
    return np.dot(test_datum, w)
    ## TODO




def run_validation(x,y,taus,val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of validation losses, one for each tau value
    '''
    ## TODO
    x_train, x_validate, y_train, y_validate = train_test_split(x, y, test_size= val_frac)
    val_loss = np.ones(len(taus))
    for i in range(0, len(taus)):
        for j in range(0, len(x_train)):
            validate = LRLS(x_train[j], x_validate, y_validate, taus[i])
            err_validate = (validate - y_validate)
            val_loss[i] = np.mean(err_validate ** 2)
    return val_loss
    ## TODO


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200)
    test_losses = run_validation(x,y,taus,val_frac=0.3)
    plt.semilogx(taus, test_losses)
    plt.xlabel("Taus")
    plt.ylabel("Loss")
    plt.show()

