import numpy as np
import scipy.io as sio
import scipy.stats as ss
import matplotlib.pyplot as plt

def verifyIndex(M, N, neighbors):
    '''
    the same as in Chapter 21
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


def gibbs_sampling(y, maxIter):
    M, N = y.shape

    # known parameters
    J = 1            # prior of W[i, j], in this sample all of the elements is 1
    noise_sigma = 2  # p(y|x) is a gaussian model with sigma=2

    data = y.ravel()
    posRV = ss.norm(1, noise_sigma)
    negRV = ss.norm(-1, noise_sigma)
    L_positive = posRV.pdf(data)  # p(y|x=+1)
    L_negative = negRV.pdf(data)  # p(y|x=-1)

    # initial values
    local_evidence = np.c_[L_negative, L_positive]   # first column is -1, the second is +1
    maxIdx = np.argmax(local_evidence, axis=1)
    X = 2 * maxIdx - 1
    X = X.reshape(y.shape)
    L_positive = L_positive.reshape(y.shape)
    L_negative = L_negative.reshape(y.shape)

    plotIter = [1, 5]
    avgX = np.zeros(X.shape)
    for i in range(maxIter):
        for ix in range(N):
            for iy in range(M):
                pos = ix, iy
                neighbors = np.array([[ix, ix-1, ix, ix+1], [iy-1, iy, iy+1, iy]])
                neighbors = verifyIndex(M, N, neighbors)
                p0 = np.exp(J * np.sum(X[neighbors])) * L_positive[pos]
                p1 = np.exp(-J * np.sum(X[neighbors])) * L_negative[pos]
                p = p0 / (p0 + p1)
                sample = np.random.rand()
                if (sample < p):
                    X[pos] = 1
                else:
                    X[pos] = -1  # Gibbs Sampling

        # no burn in
        if i + 1 in plotIter:
            subIndex = (int)('22' + str(plotIter.index(i + 1) + 2))
            titleStr = 'sample {0}, Gibbs'.format(i + 1)
            plot(subIndex, titleStr, X)

        avgX += X

    return avgX / maxIter

# load image data
data = sio.loadmat('isingImageDenoiseDemo.mat')
print(data.keys())
y = data['y']
print(y)

# plot the original noisy image
fig = plt.figure(figsize=(10, 9))
fig.canvas.set_window_title('isingImageDenoiseDmeo')

plt.subplot(221)
plt.axis('off')
plt.title('original image')
plt.imshow(y, cmap='gray', aspect='equal', interpolation='none')
plt.colorbar()

def plot(index, title, X):
    plt.subplot(index)
    plt.axis('off')
    plt.title(title)
    plt.imshow(X, cmap='gray', aspect='equal', interpolation='none')
    plt.colorbar()

maxIter = 15
finalX = gibbs_sampling(y, maxIter)

titleStr = 'mean after {0} sweeps of Gibbs'.format(maxIter)
plot(224, titleStr, finalX)

plt.tight_layout()
plt.show()