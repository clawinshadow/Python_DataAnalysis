import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from pcaFit import *

# extract data from MNIST, 关于MNIST的结构可以参考前面的pcaImageDemo.py
data = sio.loadmat('mnistAll.mat')
data = data['mnist'][0, 0]
N = 1000                  
train_all = data[0]
train_all_labels = data[2]
h, w, n = train_all.shape   # 28 * 28 * 60000
indices_3 = train_all_labels == 3
train_3 = train_all[:, :, indices_3.ravel()] # 筛选出所有为 3 的图像
X = train_3[:, :, 0:N]      # 只选1000个为 3 的训练集
X = X.reshape((h*w, N)).T   # 将28 * 28的像素压缩为一行 (1000, 784)
X = X - np.mean(X, axis=0)  # remove the mean
X_train = X[0:500]
X_test = X[500:]            # 各取一半作为训练集和测试集

# Fit model
rank = np.linalg.matrix_rank(X_train)
print('X_train rank: ', rank)
# np.unique()的一个附带功能就是按升序排好序
Ks = np.unique(np.r_[1, 5, 10, 20, np.round(np.linspace(1, 0.75*rank, 10))]).astype('int32')
print('Ks: ', Ks)

mu, V, vr, z, x_recon = PCA(X_train, np.max(Ks))  # 500 * 784

# rMSE
rmse_train = np.zeros(len(Ks))
rmse_test = np.zeros(len(Ks))
for i in range(len(Ks)):
    k = Ks[i]
    v = vr[:, :k]  # D * k
    Ztest = np.dot(X_test, v) # N * k
    X_test_recon = np.dot(Ztest, v.T) # N * D
    errTest = X_test_recon - X_test
    rmse_test[i] = (np.mean(errTest**2)) ** 0.5

    Ztrain = np.dot(X_train, v)
    X_train_recon = np.dot(Ztrain, v.T)
    errTrain = X_train_recon - X_train
    rmse_train[i] = (np.mean(errTrain**2)) ** 0.5

print(rmse_test.min(), rmse_test.max())
print(rmse_train.min(), rmse_train.max())

# PPCA log likelihood
ll_train = np.zeros(len(Ks))
ll_test = np.zeros(len(Ks))
for i in range(len(Ks)):
    k = Ks[i]
    mu2, W2, Z2, X_recon2, sigma2 = PPCA(X_train, k)
    ll_train[i] = -1 * LogL_PPCA(X_train, mu2, W2, sigma2)
    ll_test[i] = -1 * LogL_PPCA(X_test, mu2, W2, sigma2)

print(ll_test)
print(ll_train)

# profile log likelihood, use eigen values as λ
def GetEigenVals(X):
    N = len(X)
    x = X - np.mean(X, axis=0)
    S = np.dot(x.T, x) / N
    eigVals = np.real(sl.eigvals(S))
    
    return np.sort(eigVals)[::-1]

def GetProfileLL(eigVals, k):
    N = len(eigVals)
    assert k > 0 and k < N  # k的取值范围为 [1, N - 1]
    part_1 = eigVals[:k]
    part_2 = eigVals[k:]
    mu_1 = np.mean(part_1)
    mu_2 = np.mean(part_2)
    sigma2 = (np.sum((part_1 - mu_1)**2) + np.sum((part_2 - mu_2)**2)) / N
    rv1 = ss.norm(mu_1, sigma2**0.5)
    rv2 = ss.norm(mu_2, sigma2**0.5)
    profileLL = np.sum(rv1.logpdf(part_1)) + np.sum(rv2.logpdf(part_2))

    return profileLL

maxK = 50
KS = np.linspace(1, maxK, maxK)
eigenVals = GetEigenVals(X_train)
PLLs = np.zeros(len(KS))
for i in range(len(KS)):
    k = (int)(KS[i])
    PLLs[i] = GetProfileLL(eigenVals, k)

print(eigenVals)
print(PLLs)

# plots
# 1. RMSE of reconstruction error
fig1 = plt.figure(figsize=(12, 5))
fig1.canvas.set_window_title('pcaOverfitDemo_1')

def subplot1(index, title, data):
    plt.subplot(index)
    plt.axis([0, 400, 0, 60])
    plt.xticks(np.linspace(0, 400, 5))
    plt.yticks(np.linspace(0, 60, 7))
    plt.title(title)
    plt.xlabel('numPCs')
    plt.ylabel('rmse')
    plt.plot(Ks, data, color='midnightblue', marker='o', fillstyle='none')

subplot1(121, 'train set reconstruction error', rmse_train)
subplot1(122, 'test set reconstruction error', rmse_test)

# 2. negtive loglikelihood of PPCA
fig2 = plt.figure(figsize=(12, 5))
fig2.canvas.set_window_title('pcaOverfitDemo_2')

def subplot2(index, title, data, axis, yticks):
    ax = plt.subplot(index)
    plt.axis(axis)
    plt.xticks(np.linspace(0, 400, 5))
    plt.yticks(yticks)
    plt.title(title)
    plt.xlabel('numPCs')
    plt.ylabel('neg log lik')
    plt.text(0.01, 1.01, r'$x 10^6$', transform=ax.transAxes)
    plt.plot(Ks, data / 1e06, color='midnightblue', marker='o', fillstyle='none')
    
subplot2(121, 'train set negative loglik', ll_train, [0, 400, 1.2, 2.2], np.linspace(1.2, 2.2, 11))
subplot2(122, 'test set negative loglik', ll_test, [0, 400, 1.5, 4.5], np.arange(1.5, 4.6, 0.3))

# 3. scree plot & profile LL
fig3 = plt.figure(figsize=(12, 5))
fig3.canvas.set_window_title('pcaOverfitDemo_3')

ax = plt.subplot(121)
plt.axis([0, 50, 0, 4])
plt.xticks(np.linspace(0, 50, 6))
plt.yticks(np.linspace(0, 4, 9))
plt.title('scree plot')
plt.xlabel('numPCs')
plt.ylabel('eigen value')
plt.text(0.01, 1.01, r'$x 10^5$', transform=ax.transAxes)
plt.plot(KS, eigenVals[:maxK] / 1e05, color='midnightblue', fillstyle='none')
    
ax = plt.subplot(122)
plt.axis([0, 50, -8900, -8400])
plt.xticks(np.linspace(0, 50, 6))
plt.yticks(np.linspace(-8900, -8400, 6))
plt.xlabel('numPCs')
plt.ylabel('profile log likelihood')
plt.plot(KS, PLLs, color='midnightblue', fillstyle='none') 

plt.show()
