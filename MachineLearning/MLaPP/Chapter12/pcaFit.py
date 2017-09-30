import numpy as np
import scipy.linalg as sl
import sklearn.preprocessing as spp

def PCA(X, L=1):
    scaler = spp.StandardScaler(with_std=False).fit(X)  # 仅仅只是移除mean就够了，不用将标准差转化成1
    mu = scaler.mean_   # 重构X时要加回去的
    x = scaler.transform(X)
    N, D = x.shape
    assert D >= L
    S = np.dot(x.T, x) / N
    # Fit PPCA model, 参数包括： mu, sigma2, W
    w, vr = sl.eig(S)     # 求解S的特征值和特征向量
    sortedIndices = np.argsort(w)[::-1] # reverse w
    L_indices = sortedIndices[:L]       # top L
    w = w[L_indices]                    # top L eigvals
    vr = (vr.T[L_indices]).T            # top L eigen vectors, D * L

    # get Z and Reconstuctions
    Z = np.dot(x, vr)
    x_recon = np.dot(Z, vr.T) + mu

    return mu, vr, Z, x_recon

def PPCA(X, L=1):
    '''
    PPCA 与 PCA 的区别在于
    1. PCA是个非概率模型，PPCA是个概率模型
    2. PPCA中的 Ψ = σ2*I， 当 σ -> 0 时，就是PCA
    书中的结论中，是假设X移除了均值的，即mean(X) = 0
    '''
    scaler = spp.StandardScaler(with_std=False).fit(X)  # 仅仅只是移除mean就够了，不用将标准差转化成1
    mu = scaler.mean_   # 重构X时要加回去的
    x = scaler.transform(X)
    N, D = x.shape
    assert D >= L
    S = np.dot(x.T, x) / N
    # Fit PPCA model, 参数包括： mu, sigma2, W
    w, vr = sl.eig(S)     # 求解S的特征值和特征向量
    sortedIndices = np.argsort(w)[::-1] # reverse w
    L_indices = sortedIndices[:L]       # top L
    w = w[L_indices]                    # top L eigvals
    vr = (vr.T[L_indices]).T            # top L eigen vectors, D * L
    rest_w = sortedIndices[L:]  # 剩余的特征值
    sigma2 = np.mean(rest_w)    # MLE of σ2
    t = (np.diag(w) - sigma2 * np.eye(L)) ** 0.5
    W = np.dot(vr, t)   # D * L

    # Infer Latent Z : N * L，隐藏变量需要用mean=0的x来计算
    F = np.dot(W.T, W) + sigma2 * np.eye(L)
    Z_mean = np.dot(x, np.dot(sl.inv(F), W.T).T)  # Z 是一个随机变量，这里只给出Z的均值
    # Reconstruct X
    X_recon = np.dot(Z_mean, W.T) + mu

    return mu, W, Z_mean, X_recon

