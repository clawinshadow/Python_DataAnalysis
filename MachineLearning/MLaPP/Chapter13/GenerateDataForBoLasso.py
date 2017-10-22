import numpy as np
import scipy.io as sio
import scipy.linalg as sl
import scipy.stats as ss

def cov2cor(V):
    '''
    将一个协方差矩阵转化为相关阵，参数V已经是一个协方差矩阵了
    Sigma(i) = sqrt( Covariance(i,i) );
    Corr(i,j) = Covariance(i,j)/( Sigma(i)*Sigma(j) );
    '''
    S = np.diag(V)**0.5  # Sigma(i)
    S = S.reshape(-1, 1)
    R = V / np.dot(S, S.T)
    for i in range(len(R)):
        R[i, i] = 1      # force diagonal values equal to 1

    return R

def bolassoMakeData(n, d, r, nDS, requireConsistent=False):
    '''
    n - number of examples
    d - number of features
    r - number of relevant features
    nDS - number of DataSets
    requireConsistent - details refer to www.di.ens.fr/~fbach/icml_bolasso.pdf
    noise - standard deviation of noise added to y (default 0.1)
    '''
    noise = 0.1
    Xdata = np.zeros((nDS, n, d))
    ydata = np.zeros((nDS, n, 1))

    done = False
    maxIter = 100
    iterNo = 1
    while not done:  # keep looping until we find a distribution that matches the consistency criteria
        mu = np.zeros(d)
        G = np.random.randn(d, d)
        Q = np.dot(G, G.T)
        Q = cov2cor(Q)
        sigma = Q

        W = np.random.randn(r)
        W = W / sl.norm(W, 2)
        W = W * (2 * np.random.rand(1) / 3 + 1 / 3) # rescale by amount randomly and uniformly chosen in [1/3,1]
        W = np.r_[W, np.zeros(d - r)]

        Q1 = Q[r:d, :r]  # bottom left sub mat 8 * 8
        Q2 = Q[:r, :r]   # top left sub mat 8 * 8
        W1 = np.sign(W[:r]).reshape(-1, 1)
        m = sl.norm(np.dot(Q1, np.dot(Q2, W1)), np.inf)
        isConsistent = m <= 1
        done = (not requireConsistent) or (requireConsistent and isConsistent)
        done = done or (iterNo > maxIter)
        iterNo += 1

    if iterNo >= maxIter:
        print('fail to make consistent data')

    for i in range(nDS):
        X = ss.multivariate_normal(mu, sigma).rvs(n)
        W = W.reshape(-1, 1)
        y = np.dot(X, W)
        s = noise * np.mean(y**2)**0.5
        noises = (ss.norm(0, s).rvs(n)).reshape(-1, 1)
        y = y + noises
        Xdata[i] = X
        ydata[i] = y

    return Xdata, ydata, W

def demo():
    n = 1000
    d = 16
    r = 8
    nDS = 256
    X, y, w = bolassoMakeData(n, d, r, nDS, True)
    data = dict()
    data['X'] = X
    data['y'] = y
    data['w'] = w
    sio.savemat('bolassoData.mat', data)

    data = sio.loadmat('bolassoData.mat')
    print(data['X'].shape, data['y'].shape, data['w'].shape)
