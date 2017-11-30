import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt

'''basically it's a demo of markov chain running to stationary distribution'''

# construction transition matrix T
N = 20
T = np.zeros((N + 1, N + 1))
for i in range(N + 1):
    # left neighbour
    if i - 1 < 0:
        T[i, i] = 0.5
    else:
        T[i, i - 1] = 0.5

    # right neighbour
    if i + 1 > N:
        T[i, i] = 0.5
    else:
        T[i, i + 1] = 0.5

# 2 different initial states
p0_a = np.zeros(N + 1)
p0_a[10] = 1     # start from 10
p0_b = np.zeros(N + 1)
p0_b[17] = 1     # start from 17

# plots
iters = np.array([0, 1, 2, 3, 10, 100, 200, 400])

fig = plt.figure(figsize=(9, 11))
fig.canvas.set_window_title('randomWalk0to20Demo')

pas = np.zeros((len(iters), N + 1))
pbs = np.zeros((len(iters), N + 1))
for i in range(len(iters)):
    iter = iters[i]
    p_a = p0_a
    p_b = p0_b
    if iter > 0:
        for k in range(1, iter+1):
            p_a = np.dot(p_a, T)
            p_b = np.dot(p_b, T)

    # draw left plot
    ax = fig.add_subplot(8, 2, 2 * i + 1)
    ax.tick_params(left='off', bottom='off', labelleft='off')
    if i == 0:
        plt.title(r"$Initial Condition X_0 = 10$")
    plt.xlim([-0.5, 20.5])
    plt.xticks(np.linspace(0, 20, 5))
    plt.ylabel('p(x)_{0}'.format(iters[i]))
    plt.plot(np.linspace(0, N, N + 1), p_a.ravel(), 'o', color='darkred', linestyle='none', mec='k')
    for j in range(N + 1):
        plt.plot([j, j], [0, p_a.ravel()[j]], color='midnightblue', lw=1)

    # draw right plot
    ax2 = fig.add_subplot(8, 2, 2 * i + 2)
    ax2.tick_params(left='off', bottom='off', labelleft='off')
    if i == 0:
        plt.title(r'$Initial Condition X_0 = 17$')
    plt.xlim([-0.5, 20.5])
    plt.xticks(np.linspace(0, 20, 5))
    plt.ylabel('p(x)_{0}'.format(iters[i]))
    plt.plot(np.linspace(0, N, N + 1), p_b.ravel(), 'o', color='darkred', linestyle='none', mec='k')
    for j in range(N + 1):
        plt.plot([j, j], [0, p_b.ravel()[j]], color='midnightblue', lw=1)

plt.tight_layout()
plt.show()



