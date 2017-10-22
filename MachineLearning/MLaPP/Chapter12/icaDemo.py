import numpy as np
import sklearn.decomposition as sd
import matplotlib.pyplot as plt

'''
可参考：
http://scikit-learn.org/stable/auto_examples/
decomposition/plot_ica_blind_source_separation.html#sphx-glr-auto-examples-decomposition-plot-ica-blind-source-separation-py
'''

np.random.seed(1)

def demosig():
    D, N = 4, 500
    v = np.arange(0, 500, 1)
    print(len(v))
    sig = np.zeros((D, N))
    sig[0] = np.sin(v/2)
    sig[1] = ((np.remainder(v, 23) - 11) / 9) ** 5
    sig[2] = ((np.remainder(v, 27) - 13) / 9)
    sig[3] = ((np.random.rand(N) < 0.5) * 2 - 1) * np.log(np.random.rand(N))

    for i in range(D):
        std = (np.var(sig[i]))**0.5
        sig[i] = sig[i] / std

    sig = sig - np.mean(sig, axis=1).reshape(-1, 1) # remove mean

    mixcoef = np.random.rand(D, D)          # 在这个W中，D = L = 4
    mixedsig = np.dot(mixcoef, sig)         # 同样是 4 * N

    return v, sig, mixedsig

v, sig, mixedsig = demosig()

# fit PCA
pca = sd.PCA(n_components=4).fit(mixedsig.T)
X_PCA = pca.transform(mixedsig.T)
X_PCA = X_PCA.T
print(X_PCA.shape)

# fit ICA
# 这个与书里面不同的是 scale不一样，画图的时候yaxis让图形自己决定就好
X_ICA = sd.FastICA(n_components=4).fit_transform(mixedsig.T)
X_ICA = X_ICA.T
print(X_ICA.shape)

# plot the true signal
fig1 = plt.figure(figsize=(13, 6))
fig1.canvas.set_window_title('icaDemo_1')

def plotData(index, title, ylim, x, y, use_ylim=True):
    plt.subplot(index)
    if not title == None:
        plt.title(title)
    plt.xlim(0, 500)
    if use_ylim:
        plt.ylim(ylim)
        plt.yticks(np.linspace(ylim[0], ylim[1], 3))
    plt.xticks(np.linspace(0, 500, 6))
    plt.plot(x, y, color='midnightblue', lw=1)

plotData(421, 'truth', (-2, 2), v, sig[0])
plotData(423, None, (-5, 5), v, sig[1])
plotData(425, None, (-2, 2), v, sig[2])
plotData(427, None, (-10, 10), v, sig[3])

# plot the observations
plotData(422, 'observed signals', (-10, 10), v, mixedsig[0])
plotData(424, None, (-5, 5), v, mixedsig[1])
plotData(426, None, (-10, 10), v, mixedsig[2])
plotData(428, None, (-5, 5), v, mixedsig[3])

plt.tight_layout()

# plot the estimates
fig2 = plt.figure(figsize=(13, 6))
fig2.canvas.set_window_title('icaDemo_2')

plotData(421, 'PCA estimate', (-10, 10), v, X_PCA[0])
plotData(423, None, (-5, 5), v, X_PCA[1])
plotData(425, None, (-2, 2), v, X_PCA[2])
plotData(427, None, (-1, 1), v, X_PCA[3])

plotData(422, 'ICA estimate', None, v, X_ICA[0], False)
plotData(424, None, None, v, X_ICA[1], False)
plotData(426, None, None, v, X_ICA[2], False)
plotData(428, None, None, v, X_ICA[3], False)

plt.tight_layout()
plt.show()
