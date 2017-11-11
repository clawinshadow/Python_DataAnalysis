import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib.animation as ma

np.random.seed(4)

class hmmDemo(object):
    def __init__(self, ax, ax2, mus, covs, A, pi):
        self.ax = ax
        self.ax2 = ax2
        self.mus = mus
        self.covs = covs
        self.A = A
        self.pi = pi
        self.colors = np.array(['b', 'r', 'k'])
        self.rvs = []
        self.points = []
        self.states = []

        # base plot attributes of observations
        self.ax.tick_params(direction='in')
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-10, 20)
        self.ax.set_xticks(np.linspace(-20, 20, 9))
        self.ax.set_yticks(np.linspace(-10, 20, 7))

        # base plot attributes of hidden states
        self.ax2.tick_params(direction='in')
        self.ax2.set_xlim(0, 21)
        self.ax2.set_ylim(0.9, 3.1)
        self.ax2.set_xticks(np.linspace(2, 20, 10))
        self.ax2.set_yticks(np.linspace(1, 3, 11))

        # draw gaussians
        xx, yy = np.meshgrid(np.linspace(-20, 20, 100), np.linspace(-20, 20, 100))
        xtest = np.c_[xx.ravel(), yy.ravel()]
        for i in range(len(self.mus)):
            rv = ss.multivariate_normal(self.mus[i], self.covs[i])
            self.rvs.append(rv)
            zz = rv.pdf(xtest).reshape(xx.shape)
            zmin = zz.min()
            zmax = zz.max()
            level = zmin + 0.06 * (zmax - zmin)
            self.ax.plot(self.mus[i, 0], self.mus[i, 1], color=self.colors[i], marker='x', ms=10)
            self.ax.contour(xx, yy, zz, colors=self.colors[i], levels=[level])

    def DrawPoints(self):
        states = np.array(self.states)
        points = np.array(self.points)
        blue_points = points[states == 1]
        red_points = points[states == 2]
        black_points = points[states == 3]

        artists = []
        # draw observations
        if len(blue_points) > 0:
            artists.append(self.ax.plot(blue_points[:, 0], blue_points[:, 1], 'bo', linestyle='none'))
        if len(red_points) > 0:
            artists.append(self.ax.plot(red_points[:, 0], red_points[:, 1], 'ro', linestyle='none'))
        if len(black_points) > 0:
            artists.append(self.ax.plot(black_points[:, 0], black_points[:, 1], 'ko', linestyle='none'))

        # draw states
        artists.append(self.ax2.scatter(np.linspace(1, len(states), len(states)), \
                                     states, c=self.colors[states - 1]))

        return artists

    def GeneratePoint(self, probs):
        mb = ss.multinomial(1, probs)  # set n = 1 in multinomial to simulate an multi-bernulli
        state = np.flatnonzero(mb.rvs(1)[0])[0] + 1  # should always be 1 in this sample
        self.states.append(state)
        gaussian = self.rvs[state - 1]
        point = gaussian.rvs(1)
        self.points.append(point)

    def GetNextPoint(self):
        lastState = self.states[-1]
        self.GeneratePoint(self.A[lastState - 1])

    def init(self):
        self.GeneratePoint(self.pi)  # generate first point
        return self.DrawPoints()

    def __call__(self, i):
        self.GetNextPoint()
        return self.DrawPoints()

# initialize params
K = 3   # states of Z
D = 2   # dimension of X

mus = 10 * np.array([[-1, 0],
                     [1, 0],
                     [0, 1]])       # centers of gaussian

covs = np.zeros((3, 2, 2))          # covariances of gaussian
covs[0] = 10 * np.array([[1, 0.5],
                         [0.5, 1]])
covs[1] = 10 * np.array([[1, -0.5],
                         [-0.5, 1]])
covs[2] = 10 * np.array([[3, 0],
                         [0, 0.5]])

A = np.array([[0.8, 0.1, 0.1],
              [0.1, 0.8, 0.1],
              [0.1, 0.1, 0.8]])     # transition matrix
pi = np.array([1, 0, 0])            # probs of initial z state, absolutely start from 1
N = 19

# draw animation
fig = plt.figure(figsize=(10.5, 5))
fig.canvas.set_window_title('hmmLillypadDemo')

ax1 = plt.subplot(121)
ax2 = plt.subplot(122)

hmm = hmmDemo(ax1, ax2, mus, covs, A, pi)
anim = ma.FuncAnimation(fig, hmm, N, init_func=hmm.init, interval=500, repeat=False)

plt.tight_layout()
plt.show()