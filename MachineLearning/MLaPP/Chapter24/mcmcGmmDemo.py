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
seeds = np.array([0, 1, 2])

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

# Sampling x using Gibbs Sampling
K = len(mixweight)

def sample_discrete(probs):
    sample = ss.multinomial(1, probs).rvs(1)[0]
    return np.argmax(sample)

def sample_q(x):
    # x means observed data point, q means latent discrete variable, belongs to which cluster
    post = np.zeros(K)
    for i in range(K):
        post[i] = mixweight[i] * ss.norm.pdf(x, loc=mus[i], scale=np.sqrt(sigmas[i]))  # posterior: p(zi|xi, theta)
    post = post / np.sum(post)

    return sample_discrete(post)

def sample_x(q):
    mu = mus[q]
    sigma = np.sqrt(sigmas[q])
    x = ss.norm.rvs(loc=mu, scale=sigma, size=1)
    return x

def EPSR(samples):
    '''
    用于跑了多个chain的情况下，用来评估收敛性的一个指标，具体算法参见24.4.3.1，所以samples一般要是二维数组，每一列都代表一个chain
    '''
    n, m = samples.shape
    mean_per_chain = np.mean(samples, axis=0)
    mean_overall = np.mean(samples)
    if m > 1:
        B = (n / (m - 1)) * np.sum((mean_per_chain - mean_overall)**2)  # variance of different chains
        var_per_chain = np.var(samples, axis=0)
        W = 1/m * np.sum(var_per_chain)

        V = (n - 1) * W / n + B / n
        R = np.sqrt(V / W)
    else:
        R = np.nan

    return R

def acf(x, lagmax=40):
    # auto correlation function, 这个不怎么看得懂
    x = x - np.mean(x)
    n = len(x)
    y = np.convolve(x[::-1], x)  # 1000 + 1000 - 1
    y = y[:n]
    y = y[::-1] / n
    s2 = y[0]
    y = y / s2

    return y[:lagmax+1]

N1 = 1000
sample_gibbs = np.zeros((N1, 2))     # first column is x, the second is q
X_gibbs = np.zeros((N1, len(seeds)))
for s in range(len(seeds)):
    np.random.seed(seeds[s])
    x = mus[1] + np.random.randn()  # initial x
    q = sample_discrete(mixweight)  # initial q
    for i in range(N1):
        x = sample_x(q)
        q = sample_q(x)
        if s == 0:
            sample_gibbs[i] = [x, q]  # 存一个就可以了
        X_gibbs[i, s] = x

epsr_gibbs = EPSR(X_gibbs)
acf_gibbs = acf(X_gibbs[:, 1])

# Sampling x using Metropolis-Hasting algorithm,
# because gaussian is a symmetric proposal distribution, so don't use Hasting correction
N = 1000
xinit = mus[1] + np.random.randn()    # start from one of the modes
samples = np.zeros((len(proposal_sigmas), N, len(seeds)))
epsr_mh = np.zeros(len(proposal_sigmas))
acf_mh = np.zeros((len(proposal_sigmas), len(acf_gibbs)))
for i in range(len(proposal_sigmas)):
    p_sigma = proposal_sigmas[i]
    for k in range(len(seeds)):
        np.random.seed(seeds[k])
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

        samples[i, :, k] = X
    epsr_mh[i] = EPSR(samples[i])
    acf_mh[i] = acf(samples[i, :, 0])

# plots
fig = plt.figure(figsize=(10, 9))
fig.canvas.set_window_title('mcmcGmmDemo')

def plot3d(index, data, sigma):
    ax = fig.add_subplot(2, 2, index, projection='3d')
    titleStr = r'$MH_with_N(0, {0:.3f}^2)_proposal$'.format(sigma)

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

# plot gibbs sampling
ax = plt.subplot(221)
ax.tick_params(direction='in')
colors = ['r', 'b']
plt.title('Gibbs Sampling')
plt.xlim(0, N1)
plt.xticks(np.linspace(0, N1, 6))
diffs = np.unique(sample_gibbs[:, 1])
for i in range(len(diffs)):
    idx = sample_gibbs[:, 1] == diffs[i]
    points = sample_gibbs[idx, 0]
    xvals = np.flatnonzero(idx)  # gain all the indices that is True
    plt.plot(xvals, points, 'o', color=colors[i], fillstyle='none', linestyle='none')

# plot MH sampling
for i in range(len(proposal_sigmas)):
    plot3d(i+2, samples[i, :, 0], proposal_sigmas[i])

plt.tight_layout()

# plot trace plot
fig2 = plt.figure(figsize=(10, 9))
fig2.canvas.set_window_title('mcmcGmmDemo_trace_plot')

def trace_plot(index, title, data):
    ax = plt.subplot(index)
    ax.tick_params(direction='in')
    N, D = data.shape
    plt.title(title)
    plt.xlim(0, N)
    plt.xticks(np.linspace(0, N, 6))
    plt.plot(np.linspace(1, N, N), data, '-', lw=0.5)

titleStr = 'gibbs, Rhat = {0:.4f}'.format(epsr_gibbs)
trace_plot(224, titleStr, X_gibbs)
for i in range(len(proposal_sigmas)):
    titleStr = r'$MH-N(0, {0:.3f}^2), R-hat = {1:.4f}$'.format(proposal_sigmas[i], epsr_mh[i])
    index = 220 + i + 1
    trace_plot(index, titleStr, samples[i])
plt.tight_layout()

# plot Autocorrelation stem plot, 这个与书里面的差距比较大，暂时查不出来原因
fig3 = plt.figure(figsize=(10, 9))
fig3.canvas.set_window_title('mcmcGmmDemo_acf_plot')

def stem_plot(index, title, data):
    ax = plt.subplot(index)
    ax.tick_params(direction='in')
    plt.xlim(0, 45)
    plt.xticks(np.linspace(0, 45, 10))
    if index in [223, 224]:
        plt.ylim(-0.2, 1.2)
        plt.yticks(np.linspace(-0.2, 1.2, 8))
    else:
        plt.ylim(0, 1)
        plt.yticks(np.linspace(0, 1, 11))
    plt.title(title)
    plt.axhline(0, color='k', lw=0.5)
    N = len(data)
    plt.plot(np.linspace(1, N, N), data, 'o', fillstyle='none', color='midnightblue', linestyle='none', mew=0.5)
    for i in range(N):
        plt.plot([i+1, i+1], [0, data[i]], '-', color='midnightblue', lw=0.5)

titleStr = 'gibbs'
stem_plot(224, titleStr, acf_gibbs)
for i in range(len(proposal_sigmas)):
    titleStr = r'$MH-N(0, {0:.3f}^2)$'.format(proposal_sigmas[i])
    index = 220 + i + 1
    stem_plot(index, titleStr, acf_mh[i])

plt.tight_layout()
plt.show()
