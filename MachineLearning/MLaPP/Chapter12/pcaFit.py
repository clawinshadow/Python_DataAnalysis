import numpy as np
import scipy.linalg as sl
import scipy.stats as ss
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
    vr2 = (vr.T[sortedIndices]).T        # ordered eigen vectors
    L_indices = sortedIndices[:L]       # top L
    w = w[L_indices]                    # top L eigvals
    vr3 = (vr.T[L_indices]).T            # top L eigen vectors, D * L

    # get Z and Reconstuctions
    Z = np.dot(x, vr3)
    x_recon = np.dot(Z, vr3.T) + mu

    return mu, vr2, vr3, Z, x_recon

def PPCA(X, L=1):
    '''
    PPCA 与 PCA 的区别在于
    1. PCA是个非概率模型，PPCA是个概率模型
    2. PPCA中的 Ψ = σ2*I， 当 σ -> 0 时，就是PCA
    3. PCA中的W是orthonormal的，正交规范化的集，即W中的列向量除了彼此正交外，还是单位向量
       PPCA中的W是orthogonal的，只要彼此正交就可以
    书中的结论中，是假设X移除了均值的，即mean(X) = 0
    '''
    scaler = spp.StandardScaler(with_std=False).fit(X)  # 仅仅只是移除mean就够了，不用将标准差转化成1
    mu = scaler.mean_   # 重构X时要加回去的
    x = scaler.transform(X)
    N, D = x.shape
    assert D >= L
    S = np.dot(x.T, x) / N
    # Fit PPCA model, 参数包括： mu, sigma2, W
    w_all, vr_all = sl.eig(S)     # 求解S的特征值和特征向量

    sortedIndices = np.argsort(w_all)[::-1] # reverse w
    L_indices = sortedIndices[:L]           # top L
    w = w_all[L_indices]                    # top L eigvals
    vr = (vr_all.T[L_indices]).T            # top L eigen vectors, D * L
    rest_w = w_all[sortedIndices[L:]]       # 剩余的特征值
    sigma2 = np.mean(rest_w)                # MLE of σ2
    print('sigma2: ', sigma2)
    t = (np.diag(w) - sigma2 * np.eye(L)) ** 0.5
    W = np.dot(vr, t)           # D * L，它已经是正交的了，书中的R实际上是任意一个Rotation Matrix

    # Infer Latent Z : N * L，隐藏变量需要用mean=0的x来计算
    F = np.dot(W.T, W) + sigma2 * np.eye(L)
    FW = np.dot(sl.inv(F), W.T)   # 最终的 x -> z 的loading matrix，不再是正交矩阵
    print('sl.norm(FW): ', sl.norm(FW[:, 0]))  
    Z_mean = np.dot(x, FW.T)      # Z 是一个随机变量，这里只给出Z的均值
    
    # Reconstruct X
    X_recon = np.dot(Z_mean, W.T) + mu  # 重构X时，矩阵还是用W，而不是用FW，这是PPCA的定义

    return mu, W, Z_mean, X_recon, sigma2
    
def LogL_PPCA(X, mu, W, sigma2):
    N, D = X.shape
    cov = np.dot(W, W.T) + sigma2 * np.eye(D)  # D * D
    rv = ss.multivariate_normal(mu, cov)
    
    return np.sum(rv.logpdf(X))

