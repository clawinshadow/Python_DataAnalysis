import numpy as np

'''
Viterbi algorithm, used for seeking the most probable state chain argmax(p(Z_1:T | X_1:T))
three two steps:
1. generate delta & a matrix
2. trace back 
'''

def normalize(a):
    return a / np.sum(a)

def hmm_viterbi(X, A, obsModel, pi):
    '''
    :param X: observations
    :param A: transition matrix
    :param obsModel: P(x | z = j)
    :param pi: initial distribution of states
    :return: Z, with equal length of X, the most probable state chain
    '''
    N = len(X)
    assert N > 0
    pi = pi.reshape(-1, 1)
    delta = np.zeros((N, 2))
    a = np.zeros(delta.shape)
    Z = np.zeros(N)

    phi1 = np.array([obsModel[0, X[0] - 1], obsModel[1, X[0] - 1]])  # local evidence P(x1 | z = j)
    phi1 = phi1.reshape(-1, 1)
    delta1 = normalize(phi1 * pi)
    delta[0] = delta1.ravel()
    Z[0] = np.argmax(delta[0])

    # generate delta and a matrix
    for i in range(1, N):
        xi = X[i]
        pre_delta = delta[i - 1]
        delta_i = np.zeros(pre_delta.shape)
        for j in range(len(pre_delta)):
            tmp = []
            phi_j = obsModel[j, xi - 1]
            for k in range(len(pre_delta)):
                tmp.append(pre_delta[k] * A[k, j] * phi_j)

            delta_i[j] = np.max(tmp)
            a[i, j] = np.argmax(tmp)

        delta[i] = normalize(delta_i)   # to avoid underflow

    Z[-1] = np.argmax(delta[-1])
    # trace back to seek state chain
    for i in range(N - 2, 0, -1):
        next_z = Z[i + 1]
        next_a = a[i + 1]
        Z[i] = next_a[(int)(next_z)]

    return Z.ravel()