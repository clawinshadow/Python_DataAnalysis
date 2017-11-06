import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

np.random.seed(1)

def sample(x, aw1, ab1, aw2, ab2):
    '''
    2-layer forward neural network, 1 input, 12 nodes in hidden layer, 1 output
    then len(w1) = 12, len(b1) = 12, len(w2) = 12, len(b2) = 1
    :param x:   input, N * 1
    :param aw1: precision of w1 prior
    :param ab1: precision of b1 prior
    :param aw2: precision of w2 prior
    :param ab2: precision of b2 prior
    :return:    output

    simulate a forward process, from input -> output, using tanh in hidden layer and linear output function
    '''
    mu1 = np.zeros(12)
    mu2 = np.zeros(12)
    mu3 = np.zeros(12)
    mu4 = 0
    cov1 = np.eye(12) / aw1
    cov2 = np.eye(12) / ab1
    cov3 = np.eye(12) / aw2
    cov4 = 1 / ab2
    N_samples = 10
    w1 = ss.multivariate_normal(mu1, cov1).rvs(N_samples)
    b1 = ss.multivariate_normal(mu2, cov2).rvs(N_samples)
    w2 = ss.multivariate_normal(mu3, cov3).rvs(N_samples)
    b2 = ss.norm(mu4, np.sqrt(cov4)).rvs(N_samples)

    N, D = x.shape
    y = np.zeros((N_samples, N))
    for i in range(N_samples):
        w1_, b1_ = w1[i].reshape(1, -1), b1[i].reshape(1, -1)
        w2_, b2_ = w2[i].reshape(-1, 1), b2[i]
        # from input -> hidden layer
        z = np.tanh(np.dot(x, w1_) + np.dot(np.ones((N, 1)), b1_))
        # from hidder layer -> output
        y[i] = (np.dot(z, w2_) + b2_).ravel()  # N * 1

    print(y.shape)
    return y

xs = np.linspace(-1, 1, 401).reshape(-1, 1)
w1s = [0.01, 0.001, 0.01, 0.01, 0.01]
b1s = [0.1, 0.1, 0.01, 0.1, 0.1]
w2s = [1.0, 1.0, 1.0, 0.1, 1.0]
b2s = [1.0, 1.0, 1.0, 1.0, 0.1]

# plots
fig = plt.figure(figsize=(13, 8))
fig.canvas.set_window_title('mlpPriorsDemo')

for i in range(len(w1s)):
    index = (int)('23' + str(i + 1))
    y = sample(xs, w1s[i], b1s[i], w2s[i], b2s[i])
    plt.subplot(index)
    plt.title(r'$\alpha_w1={0}, \alpha_b1={1}, \alpha_w2={2}, \alpha_b2={3}$'.format( \
        w1s[i], b1s[i], w2s[i], b2s[i]))
    plt.axis([-1, 1, -10, 10])
    plt.xticks(np.linspace(-1, 1, 5))
    plt.yticks(np.linspace(-10, 10, 11))
    plt.plot(xs.ravel(), y.T, color='k')

plt.tight_layout()
plt.show()

