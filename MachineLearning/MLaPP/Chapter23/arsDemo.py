import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

'''
Adaptive Rejection Sampling, ARS, 是rejection sampling的一个特殊形式，只能用于log concave的概率分布p(x)

half gaussian是有问题的，matlab的code一样报错，后续待查
'''

def arsComputeHulls(S, fS, domain):
    N = len(S)
    lowerHull = []
    for i in range(N - 1):
        lh = dict()
        lh['m'] = (fS[i + 1] - fS[i]) / (S[i + 1] - S[i])    # 近似点S[i]处的切线的斜率，即导数
        lh['b'] = fS[i] - lh['m'] * S[i]  # 截距
        lh['left'] = S[i]
        lh['right'] = S[i + 1]
        lowerHull.append(lh)

    upperHull = []
    uh = dict()
    uh['m'] = (fS[1] - fS[0]) / (S[1] - S[0])    # left boundary 处的导数
    uh['b'] = fS[0] - uh['m'] * S[0]             # 截距
    uh['left'] = domain[0]
    uh['right'] = S[0]
    uh['pr'] = np.exp(uh['b']) * (np.exp(uh['m'] * S[0]) - 0) / uh['m']   # integrating from -inf, 相当于cdf
    upperHull.append(uh)

    uh2 = dict()
    uh2['m'] = (fS[2] - fS[1]) / (S[2] - S[1])
    uh2['b'] = fS[1] - uh2['m'] * S[1]
    uh2['left'] = S[0]
    uh2['right'] = S[1]
    uh2['pr'] = np.exp(uh2['b']) * (np.exp(uh2['m'] * S[1]) - np.exp(uh2['m'] * S[0])) / uh2['m']
    upperHull.append(uh2)

    for i in range(1, N - 2):
        m1 = (fS[i] - fS[i - 1]) / (S[i] - S[i - 1])
        b1 = fS[i] - m1 * S[i]

        m2 = (fS[i + 2] - fS[i + 1]) / (S[i + 2] - S[i + 1])
        b2 = fS[i + 1] - m2 * S[i + 1]

        ix = (b1 - b2) / (m2 - m1)    # two lines' intersection

        pr1 = np.exp(b1) * (np.exp(m1 * ix) - np.exp(m1 * S[i])) / m1
        uh = dict()
        uh['m'] = m1
        uh['b'] = b1
        uh['pr'] = pr1
        uh['left'] = S[i]
        uh['right'] = ix
        upperHull.append(uh)

        pr2 = np.exp(b2) * (np.exp(m2 * S[i + 1]) - np.exp(m2 * ix)) / m2
        uh = dict()
        uh['m'] = m2
        uh['b'] = b2
        uh['pr'] = pr2
        uh['left'] = ix
        uh['right'] = S[i + 1]
        upperHull.append(uh)

    uh = dict()
    uh['m'] = (fS[-2] - fS[-3]) / (S[-2] - S[-3])
    uh['b'] = fS[-2] - uh['m'] * S[-2]
    uh['left'] = S[-2]
    uh['right'] = S[-1]
    uh['pr'] = np.exp(uh['b']) * (np.exp(uh['m'] * S[-1]) - np.exp(uh['m'] * S[-2])) / uh['m']
    upperHull.append(uh)

    uh = dict()
    uh['m'] = (fS[-1] - fS[-2]) / (S[-1] - S[-2])  # right boundary 处的导数
    uh['b'] = fS[-1] - uh['m'] * S[-1]  # 截距
    uh['left'] = S[-1]
    uh['right'] = domain[1]
    uh['pr'] = np.exp(uh['b']) * (0 - np.exp(uh['m'] * S[-1])) / uh['m']  # integrating from -inf, 相当于cdf
    upperHull.append(uh)

    Z = sum(d['pr'] for d in upperHull)
    for d in upperHull:
        d['pr'] /= Z

    return lowerHull, upperHull

def arsSampleUpperHull(upperHull):
    prs = [d['pr'] for d in upperHull]
    cdf= np.cumsum(prs)

    u = np.random.rand()
    for i in range(len(upperHull)):
        if u < cdf[i]:
            break

    u = np.random.rand()

    m = upperHull[i]['m']
    b = upperHull[i]['b']
    left = upperHull[i]['left']
    right = upperHull[i]['right']

    x = np.log(u * (np.exp(m * right) - np.exp(m * left)) + np.exp(m * left)) / m
    if np.isnan(x) or np.isinf(x):
        raise ValueError('Sample of x should not be infinite or NaN')

    return x

def arsEvalHulls(x, lowerHull, upperHull):
    if x < np.min([d['left'] for d in lowerHull]):
        lhVal = -np.inf
    elif x > np.max([d['right'] for d in lowerHull]):
        lhVal = -np.inf
    else:
        for i in range(len(lowerHull)):
            left = lowerHull[i]['left']
            right = lowerHull[i]['right']

            if x >= left and x <= right:
                lhVal = lowerHull[i]['m'] * x + lowerHull[i]['b']
                break

    for j in range(len(upperHull)):
        left = upperHull[j]['left']
        right = upperHull[j]['right']

        if x >= left and x <= right:
            uhVal = upperHull[j]['m'] * x + upperHull[j]['b']
            break

    return lhVal, uhVal

def ars(func, a, b, domain, nSamples, *args):
    if domain[0] >= domain[1]:
        raise ValueError('invalid Domain')
    if a >= b or np.isinf(a) or np.isinf(b) or a < domain[0] or b > domain[1]:
        raise ValueError('invalid a or b')

    numDerivStep = 1e-3
    S = np.array([a, a+numDerivStep, b-numDerivStep, b])

    meshPoints = 4
    S = np.unique(np.r_[S[0], np.linspace(S[1], S[2], meshPoints + 1), S[3]])
    fS = func(S, *args)

    lowerHull, upperHull = arsComputeHulls(S, fS, domain)
    samples = np.zeros(nSamples)
    nSamplesActually = 0
    while True:
        x = arsSampleUpperHull(upperHull)
        if np.isnan(x) or np.isinf(x):
            continue;
        lhVal, uhVal = arsEvalHulls(x, lowerHull, upperHull)

        u = np.random.rand()

        meshChange = False
        if u <= np.exp(lhVal - uhVal):
            samples[nSamplesActually] = x
            nSamplesActually += 1
        elif u <= np.exp(func(x, *args) - uhVal):
            samples[nSamplesActually] = x
            nSamplesActually += 1

            meshChange = True
        else:
            meshChange = True

        if meshChange:
            S = np.sort(np.r_[S, x])
            fS = func(S, *args)

            lowerHull, upperHull = arsComputeHulls(S, fS, domain)

        if nSamplesActually == nSamples:
            break

    return samples

def gaussian(x, sigma):
    return np.log(np.exp(-(x**2)/sigma))

def halfGaussian(x, sigma):
    realx = np.copy(x)
    realx[realx > 0] = 0

    return np.log(np.exp(-(realx**2)/sigma))

# gaussian
a = -2
b = 2
domain = [-np.inf, np.inf]
nSamples = 20000
sigma = 1
samples = ars(gaussian, a, b, domain, nSamples, sigma)
xs = np.linspace(-3, 3, 200)
ys = np.exp(gaussian(xs, sigma))

# plot
fig = plt.figure(figsize=(10, 4))
fig.canvas.set_window_title('ARS demo of Full Gaussian')

ax1 = plt.subplot(121)
ax1.tick_params(direction='in')
plt.title('f(x) gaussian')
plt.axis([-3, 3, 0, 1])
plt.xticks(np.linspace(-3, 3, 7))
plt.yticks(np.linspace(0, 1, 6))
plt.plot(xs, ys, '-')

ax2 = plt.subplot(122)
ax2.tick_params(direction='in')
plt.title('samples from f(x) by ARS')
plt.axis([-3, 3, 0, 700])
plt.xticks(np.linspace(-3, 3, 7))
plt.yticks(np.linspace(0, 700, 8))
plt.hist(samples, bins=100, normed=False, edgecolor='k')

plt.tight_layout()
plt.show()

'''
# half gaussian
a = -2
b = 0
domain = [-np.inf, 0]
nSamples = 20000
sigma = 3
ys = np.exp(halfGaussian(xs, sigma))
samples = ars(halfGaussian, a, b, domain, nSamples, sigma)

fig2 = plt.figure(figsize=(10, 4))
fig2.canvas.set_window_title('ARS demo of half Gaussian')

ax1 = plt.subplot(121)
ax1.tick_params(direction='in')
plt.title('f(x) half gaussian')
plt.axis([-3, 3, 0, 1])
plt.xticks(np.linspace(-3, 3, 7))
plt.yticks(np.linspace(0, 1, 6))
plt.plot(xs, ys, '-')

ax2 = plt.subplot(122)
ax2.tick_params(direction='in')
plt.title('samples from f(x) by ARS')
plt.axis([-3, 3, 0, 700])
plt.xticks(np.linspace(-3, 3, 7))
plt.yticks(np.linspace(0, 700, 8))
plt.hist(samples, bins=100, normed=False, edgecolor='k')

plt.tight_layout()
plt.show()
'''