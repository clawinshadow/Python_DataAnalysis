import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

'''有充足的理由怀疑matlab中的code多除以了一个h，我写的是对的'''

def GaussianKernel(x):
    val1 = 1 / np.sqrt(2 * np.pi)
    val2 = np.exp(-x**2 / 2)

    return val1 * val2

def KDE(x, kernel, centers, h):
    N, D = x.shape
    C = len(centers)
    probMat = np.zeros((N, C))
    for i in range(N):
        xi = x[i]
        if kernel == 'gaussian':
            probMat[i] = (1/h) * GaussianKernel((xi - centers) / h)
        elif kernel == 'uniform':
            # uniform pdf
            probMat[i] = (1/h) * (np.abs(2 * (xi - centers) / h) <= 1)

    return np.mean(probMat, axis=1)

centroids = np.array([-2.1, -1.3, -0.4, 1.9, 5.1, 6.2])
hs = [1.0, 2.0]
x = np.linspace(-5, 10, 1501).reshape(-1, 1)
y1 = KDE(x, 'uniform', centroids, hs[0])
y2 = KDE(x, 'uniform', centroids, hs[1])
y3 = KDE(x, 'gaussian', centroids, hs[0])
y4 = KDE(x, 'gaussian', centroids, hs[1])

print(np.sum(y1), np.sum(y2), np.sum(y3), np.sum(y4))

# plots
fig = plt.figure(figsize=(9, 8))
fig.canvas.set_window_title('parzenWindowDemo2')

ax = plt.subplot(221)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title('unif, h=' + str(hs[0]))
plt.axis([-5, 10, 0, 0.35])
plt.xticks(np.linspace(-5, 10, 4))
plt.yticks(np.linspace(0, 0.35, 8))
plt.plot(centroids, 0.01 * np.ones(len(centroids)), 'kx', linestyle='none', ms=10)
plt.plot(x.ravel(), y1, color='midnightblue', lw=1.5)

ax = plt.subplot(222)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title('unif, h=' + str(hs[1]))
plt.axis([-5, 10, 0, 0.25])
plt.xticks(np.linspace(-5, 10, 4))
plt.yticks(np.linspace(0, 0.25, 6))
plt.plot(centroids, 0.01 * np.ones(len(centroids)), 'kx', linestyle='none', ms=10)
plt.plot(x.ravel(), y2, color='midnightblue', lw=1.5)

ax = plt.subplot(223)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title('gauss, h=' + str(hs[0]))
plt.axis([-5, 10, 0, 0.16])
plt.xticks(np.linspace(-5, 10, 4))
plt.yticks(np.linspace(0, 0.16, 9))
plt.plot(centroids, 0.01 * np.ones(len(centroids)), 'kx', linestyle='none', ms=10)
plt.plot(x.ravel(), y3, color='midnightblue', lw=1)
for i in range(len(centroids)):
    mu = centroids[i]
    probs = 0.1 * ss.norm(mu, hs[0]).pdf(x.ravel())
    plt.plot(x.ravel(), probs, 'r:')

ax = plt.subplot(224)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title('gauss, h=' + str(hs[1]))
plt.axis([-5, 10, 0, 0.12])
plt.xticks(np.linspace(-5, 10, 4))
plt.yticks(np.linspace(0, 0.12, 7))
plt.plot(centroids, 0.01 * np.ones(len(centroids)), 'kx', linestyle='none', ms=10)
plt.plot(x.ravel(), y4, color='midnightblue', lw=1)
for i in range(len(centroids)):
    mu = centroids[i]
    probs = 0.1 * ss.norm(mu, hs[1]).pdf(x.ravel())
    plt.plot(x.ravel(), probs, 'r:')

plt.tight_layout()
plt.show()

