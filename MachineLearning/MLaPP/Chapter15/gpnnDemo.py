import numpy as np
import scipy.linalg as sl
import scipy. stats as ss
import matplotlib.pyplot as plt

np.random.seed(2)

def kNN(x, stda, stdb):
    '''restrict x to be 1-D array, so need 2 variances, one for x and the other for bias'''
    x = x.reshape(-1, 1)
    N, D = x.shape
    vara, varb = stda**2, stdb**2
    assert D == 1
    gram = np.zeros((N, N))
    for i in range(N):
        xi = x[i]
        for j in range(N):
            xj = x[j]
            cij = vara + xi * varb * xj
            cii = vara + xi * varb * xi
            cjj = vara + xj * varb * xj
            val = 2 * cij / (np.sqrt((1 + 2 * cii) * (1 + 2 * cjj)))
            gram[i, j] = (2 / np.pi) * np.arcsin(val)

    return gram

N = 201
xvec = np.linspace(-5, 5, N)
S = kNN(xvec, 10, 10)
print(S)
xx, yy = np.meshgrid(xvec, xvec)

stdbs = [10, 3, 1]
ytests = np.zeros((3, N))
for i in range(len(stdbs)):
    Si = kNN(xvec, 10, stdbs[i])
    varnoi = 1e-10
    cov = Si + varnoi * np.eye(N)
    rvs = ss.multivariate_normal(np.zeros(N), cov, allow_singular=True).rvs(1)
    ytests[i] = rvs.ravel()

# plots
fig = plt.figure(figsize=(10.5, 5))
fig.canvas.set_window_title('gpnnDemo')

plt.subplot(121)
plt.axis([-5, 5, -5, 5])
plt.xlabel('input, x')
plt.ylabel('input, x\'')
plt.xticks(np.linspace(-4, 4, 3))
plt.yticks(np.linspace(-4, 4, 3))
CS = plt.contour(yy, xx, S, levels=[-0.5, 0, 0.5, 0.95], colors=['b', 'c', 'orange', 'brown'])
plt.clabel(CS, inline=1, fontsize=10)  # add labels on contour

plt.subplot(122)
plt.axis([-5, 5, -2, 1.5])
plt.xlabel('input, x')
plt.ylabel('output, f(x)')
plt.xticks(np.linspace(-4, 4, 3))
plt.yticks(np.linspace(-1, 1, 3))
plt.plot(xvec, ytests[0], 'k-', label=r'$\sigma = {0}$'.format(stdbs[0]))
plt.plot(xvec, ytests[1], 'k--', label=r'$\sigma = {0}$'.format(stdbs[1]))
plt.plot(xvec, ytests[2], 'k-.', label=r'$\sigma = {0}$'.format(stdbs[2]))

plt.legend()
plt.tight_layout()
plt.show()