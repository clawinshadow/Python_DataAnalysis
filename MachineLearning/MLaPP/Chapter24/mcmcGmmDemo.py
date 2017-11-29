import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''running spend time: about 30 seconds'''

np.random.seed(1)

# known model params
mixweight = np.array([0.3, 0.7])
mus = np.array([-20, 20])
sigmas = np.array([100, 100])
proposal_sigmas = np.array([1, 500, 8])

def mixGaussianPdf(x, mus=mus, sigmas=sigmas, mixweights=mixweight):
    K = len(mixweights)
    if np.isscalar(x):
        N = 1
    else:
        N = len(x)
    pdf = np.zeros((N, K))
    for i in range(K):
        mu = mus[i]
        sigma = sigmas[i]
        weight = mixweights[i]
        pdf[:, i] = weight * ss.norm(mu, np.sqrt(sigma)).pdf(x)

    return np.sum(pdf, axis=1)

xs = np.linspace(-100, 100, 1000)
ys = mixGaussianPdf(xs)

# Sampling x using Metropolis-Hasting algorithm,
# because gaussian is a symmetric proposal distribution, so don't use Hasting correction
N = 5000
xinit = mus[1] + np.random.randn()    # start from one of the modes
samples = np.zeros((N, len(proposal_sigmas)))
for i in range(len(proposal_sigmas)):
    p_sigma = proposal_sigmas[i]
    X = np.zeros(N)
    for j in range(N):
        if j == 0:
            x_prev = xinit
        else:
            x_prev = X[j - 1]

        x_next = ss.norm.rvs(loc=x_prev, scale=p_sigma, size=1)
        alpha = mixGaussianPdf(x_next) / mixGaussianPdf(x_prev)
        r = np.min([1, alpha])
        u = np.random.rand()
        if u < r:
            X[j] = x_next
        else:
            X[j] = x_prev

    samples[:, i] = X

# plots
fig = plt.figure(figsize=(10, 9))
fig.canvas.set_window_title('mcmcGmmDemo')

def plot3d(index, data, sigma):
    ax = fig.add_subplot(2, 2, index, projection='3d')
    titleStr = r'$MH  with  N(0, {0:.3f}^2) proposal$'.format(sigma)

    # base plot attributes
    ax.set_title(titleStr, fontdict={'fontsize': 10})
    ax.set_xlim(0, N)
    ax.set_ylim(-100, 100)
    ax.set_xticks(np.linspace(0, N, 6))
    ax.set_yticks(np.linspace(-100, 100, 5))
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Samples')
    ax.w_xaxis.set_pane_color([1.0, 1.0, 1.0, 1.0])  # set background color
    ax.w_yaxis.set_pane_color([1.0, 1.0, 1.0, 1.0])
    ax.w_zaxis.set_pane_color([1.0, 1.0, 1.0, 1.0])
    ax.view_init(elev=30., azim=-30)

    # draw samples plot
    n_iter = len(data)
    ax.plot(np.linspace(1, n_iter, n_iter), data, zs=0, zdir='z', lw=2)  # on x-y pane

    # true pdf
    ax.plot(xs, ys, zs=0, zdir='x', color='r', lw=2)  # use zdir to indicate drawing on which 2D pane

    # sample pdf
    Nbins = 100
    hist, edges = np.histogram(data, np.linspace(-100, 100, Nbins))
    probs = (Nbins / 200) * (hist / N)
    ax.set_zlim(-0.001, 1.1 * probs.max())  # set zlim dynamically
    ax.plot(edges[:-1], probs, zs=0, zdir='x', lw=2)

for i in range(len(proposal_sigmas)):
    plot3d(i+2, samples[:, i], proposal_sigmas[i])

plt.tight_layout()
plt.show()
