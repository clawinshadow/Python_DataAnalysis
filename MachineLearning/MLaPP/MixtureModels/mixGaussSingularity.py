import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

# define GMM
mu = [0.5, 0.15]
sigma = np.power([0.12, 0.0003], 0.5)
rv1 = ss.norm(mu[0], sigma[0])
rv2 = ss.norm(mu[1], sigma[1])

def GMM(x):
    return 0.5 * rv1.pdf(x) + 0.5 * rv2.pdf(x)

x = np.arange(0, 1, 0.001)
points = np.array([0.15, 0.21, 0.25, 0.32, 0.45, 0.58, 0.72, 0.88])

# plots
fig = plt.figure(figsize=(7, 7))
fig.canvas.set_window_title('mixGaussSingularity')

plt.xlim(0, 1)
plt.ylim(0, 12)
plt.xticks([])
plt.yticks([])
plt.xlabel('X')
plt.ylabel('P(X)')
plt.plot(x, GMM(x), 'r-', lw=3)
plt.plot(points, np.zeros(len(points)), 'ko', ms=5, ls='none')
for i in range(len(points)):
    plt.vlines(points[i], 0, GMM(points[i]), 'g', lw=3)

plt.show()
