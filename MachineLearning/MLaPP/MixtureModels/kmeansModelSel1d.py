import math
import numpy as np
import scipy.stats as ss
import scipy.special as ssp
import sklearn.datasets as sd
import sklearn.mixture as sm
import sklearn.cluster as sc
import matplotlib.pyplot as plt

np.random.seed(0)

def MSE(X_test, res):
    mus = res.cluster_centers_
    labels = res.labels_
    K = len(mus)
    X_reconstruct = np.zeros((X_test.shape[0], K))  # N * K
    
    for i in range(K):
        mu = mus[i]
        X_reconstruct[:, i] = (X_test.ravel() - mu)**2

    X_reconstruct = np.min(X_reconstruct, axis=1) # N * 1
    mse = np.sum(X_reconstruct) / len(X_reconstruct)

    return mse

def NLL_density(X_test, x_sample, model):
    pi = model.weights_
    mu = model.means_
    sigma = model.covariances_
    K = len(sigma)
    N = len(X_test)
    probs = np.zeros((N, K))
    probs_2 = np.zeros((len(x_sample), K))
    for i in range(K):
        probs[:, i] = math.log(pi[i]) + ss.multivariate_normal(mu[i], sigma[i], allow_singular=True).logpdf(X_test)
        probs_2[:, i] = math.log(pi[i]) + ss.multivariate_normal(mu[i], sigma[i], allow_singular=True).logpdf(x_sample)
        
    total = ssp.logsumexp(probs, axis=1)
    NLL = -np.sum(total)
    density = np.sum(np.exp(probs_2), axis=1)
    print(density.shape, np.min(density), np.max(density))

    return NLL, density

# generate data
centers = np.array([-1, 0, 1]).reshape(-1, 1)
covs = np.tile(0.1**0.5, 3).reshape(-1, 1)
N = 1000
X_train, y_train = sd.make_blobs(N, 1, centers, covs)
X_test, y_test = sd.make_blobs(N, 1, centers, covs)
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
print(np.allclose(X_train, X_test))

# Fit with K-Means and GMM(EM)
Ks = np.array([2, 3, 4, 5, 6, 10, 15])
MSEs = np.zeros(len(Ks))
NLLs = np.zeros(len(Ks))
MUs = []
x_sample = np.linspace(-2, 2, 200)
densities = np.zeros((len(Ks), 200))
for i in range(len(Ks)):
    # K-Means
    resi = sc.KMeans(Ks[i], 'random').fit(X_train) 
    MSEs[i] = MSE(X_test, resi)
    MUs.append(resi.cluster_centers_)
    # GMM
    modeli = sm.GaussianMixture(Ks[i], 'full').fit(X_train)
    NLLs[i], densities[i] = NLL_density(X_test, x_sample, modeli)

print("MSEs: ", MSEs)
print('NLLs: ', NLLs)

# plots
fig1 = plt.figure(figsize=(11, 5))
fig1.canvas.set_window_title('MSE')

plt.subplot(121)
plt.title('Xtrain')
plt.axis([-3, 3, 0, 60])
plt.xticks(np.linspace(-3, 3, 7))
plt.yticks(np.linspace(0, 60, 7))
bins = np.arange(-2, 2, 0.1)
plt.hist(X_train, bins, color='darkblue', edgecolor='k', linewidth=0.5)

plt.subplot(122)
plt.title('MSE on test vs K for K−means')
plt.axis([1, 16, 0, 0.25])
plt.xticks(np.arange(2, 17, 2))
plt.yticks(np.arange(0, 0.26, 0.05))
plt.plot(Ks, MSEs, color='darkblue', marker='o', fillstyle='none')

fig2 = plt.figure(figsize=(12, 8))
fig2.canvas.set_window_title('centers')
def Plot(index, K, MSE, xlim, mu):
    plt.subplot(index)
    plt.title('K = {0}, mse = {1:.4}'.format(K, MSE))
    plt.axis([xlim[0], xlim[1], 0, 1])
    plt.xticks([xlim[0], 0, xlim[1]])
    plt.yticks(np.linspace(0, 1, 6))
    plt.vlines(mu, 0, 1, colors='r', linewidth=3)
Plot(231, Ks[0], MSEs[0], [-1, 1], MUs[0])
Plot(232, Ks[1], MSEs[1], [-2, 2], MUs[1])
Plot(233, Ks[2], MSEs[2], [-2, 2], MUs[2])
Plot(234, Ks[3], MSEs[3], [-2, 2], MUs[3])
Plot(235, Ks[4], MSEs[4], [-2, 2], MUs[4])
Plot(236, Ks[5], MSEs[5], [-2, 2], MUs[5])

fig3 = plt.figure()
fig3.canvas.set_window_title('NLL')
plt.title('NLL on test set vs K for GMM')
plt.axis([1, 16, 1160, 1225])
plt.xticks(np.linspace(2, 16, 8))
plt.yticks(np.arange(1160, 1225, 5))
plt.plot(Ks, NLLs, color='darkblue', marker='o', fillstyle='none')

fig4 = plt.figure(figsize=(10, 6.6))
fig4.canvas.set_window_title('Density')
def PlotDensity(index, K, NLL, x, density):
    plt.subplot(index)
    plt.title('K = {0}, NLL = {1:.4}'.format(K, NLL))
    plt.axis([-2, 2, 0, 0.5])
    plt.xticks([-2, 0, 2])
    plt.yticks(np.linspace(0, 0.5, 6))
    plt.plot(x, density, color='darkblue')

for i in range(len(Ks) - 1):
    index = (int)('23' + str(i+1))
    PlotDensity(index, Ks[i], NLLs[i], x_sample, densities[i])

plt.show()
