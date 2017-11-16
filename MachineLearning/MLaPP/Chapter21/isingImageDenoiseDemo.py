import numpy as np
import scipy.io as sio
import scipy.stats as ss
import matplotlib.pyplot as plt

def verifyIndex(M, N, neighbors):
    '''
    pay attention to the fancy indexing, assume A is a 2-D array: np,eye(4), and if we want to take 3 elements, whose
    index is (2, 2), (1, 0), (3, 3), then how can we represent the indices?
    it should be i = [[2, 1, 3], [2, 0, 3]], then A[i] = [1, 0, 1]
    the first row is all the x indices, and the second row in all the y indices

    :param M: row bound
    :param N: column bound
    :param neighbors: indices matrix
    :return:
    '''
    r, c = neighbors.shape
    assert r == 2
    is_valid = np.ones(c, dtype='bool')
    for i in range(c):
        index = neighbors[:, i]
        row, column = index[0], index[1]
        if row < 0 or row > M - 1:
            is_valid[i] = False
        if column < 0 or column > N - 1:
            is_valid[i] = False

    return list(neighbors[:, is_valid])  # must convert to list to be a valid index array


def meanfield(y, maxIter):
    M, N = y.shape

    # known parameters
    rate = 0.5       # damped updates
    J = 1            # prior of W[i, j], in this sample all of the elements is 1
    noise_sigma = 2  # p(y|x) is a gaussian model with sigma=2

    data = y.ravel()
    posRV = ss.norm(1, 2)
    negRV = ss.norm(-1, 2)
    L_positive = posRV.logpdf(data)  # Li+
    L_negative = negRV.logpdf(data)  # Li-
    logodds = (L_positive - L_negative).reshape(y.shape) # Li+ - Li-

    # initial values
    p1 = 1 / (1 + np.exp(-logodds))
    mu = 2 * p1 - 1

    for i in range(maxIter):
        mu_new = np.copy(mu)
        for ix in range(N):
            for iy in range(M):
                pos = ix, iy
                neighbors = np.array([[ix, ix-1, ix, ix+1], [iy-1, iy, iy+1, iy]])
                neighbors = verifyIndex(M, N, neighbors)
                Sbar = J * np.sum(mu[neighbors])
                mu_new[pos] = (1 - rate) * mu[pos] + rate * np.tanh(Sbar + 0.5 * logodds[pos])

        mu = mu_new

    return mu

# load image data
data = sio.loadmat('isingImageDenoiseDemo.mat')
print(data.keys())
y = data['y']
print(y)

# Fit with mean-field
M, N = y.shape
iters = [1, 3, 15]
mus = np.zeros((len(iters), M, N))
for i in range(len(iters)):
    mus[i] = meanfield(y, iters[i])

# plot the original noisy image
fig = plt.figure(figsize=(10, 9))
fig.canvas.set_window_title('isingImageDenoiseDmeo')

plt.subplot(221)
plt.axis('off')
plt.title('original image')
plt.imshow(y, cmap='gray', aspect='equal', interpolation='none')
plt.colorbar()

def plot(index, title, mu):
    plt.subplot(index)
    plt.axis('off')
    plt.title(title)
    plt.imshow(mu, cmap='gray', aspect='equal', interpolation='none')
    plt.colorbar()

plot(222, 'sample 1, meanfieldH', mus[0])
plot(223, 'sample 3, meanfieldH', mus[1])
plot(224, 'mean after 15 sweeps of meanfieldH', mus[2])

plt.tight_layout()
plt.show()