import numpy as np
import scipy.linalg as sl
import scipy.stats as ss
import scipy.special as sp

'''
本章详述如何用VBEM求解mix gaussian的问题， θ 包括 {π, μ, Λ}

首先，mix gaussian的似然函数：

     p(Z, X|θ) = Πi Πk[π_ik * N(xi|μk, Λk^-1)], i ~ [1, N], k ~ [1, K]
     
根据这个似然函数，我们设定一个共轭的先验分布：

     p(θ) = Dir(π|α0) * Πk[N(μk|m0,(β0*Λk)^-1) * Wishart(Λk|W0, ν0)]
     所以先验分布中，α0 是一个K维向量，m0是D维向量，与x的维度一致，β0和ν0是标量，L0是D*D矩阵
     
VB： p(θ, z1:N |D) ≈ q(θ) * Πi[q(zi)]

经过一些数学处理，我们可以得出下面的公式：

     q(z|θ) = Πi[Cat(zi|ri)]， 每一个ri都是一个K维的向量，代表着每个数据点在每个cluster中的权重，是不是很像普通EM中的E步
     q(θ) = Dir(π|α) * Πk[N(μk|mk,(βkΛk)^-1) * Wi(Λk|Wk,νk)]
     展现在图形里的时候，sparsity根据α来决定，形状根据每个mk，βk，... 等参数来决定
     
具体更新的公式就不写了，太复杂，可参考书里面的公式
'''

def mixGaussBayesStructure(alpha, beta, m, v, W):
    params = {}
    params['alpha'] = alpha   # should be K * 1
    params['beta'] = beta     # should be K * 1
    params['m'] = m           # should be K * D
    params['v'] = v           # should be K * 1
    params['W'] = W           # should be K * D * D
    K, D = m.shape
    params['invW'] = np.zeros(W.shape)
    for i in range(K):
        params['invW'][i] = sl.inv(W[i])

    # precompute for speed performance
    params['logPiTilde'] = sp.digamma(alpha) - sp.digamma(np.sum(alpha))          # E[ln(πk)]
    logdetW = np.zeros(K)
    params['logLambdaTilde'] = np.zeros((K, 1))                                        # E[ln(Λk)]
    params['entropy'] = np.zeros((K, 1))                                               # H[q(Λk)], for calculate lower bound
    params['logDirConst'] = sp.gammaln(np.sum(alpha)) - np.sum(sp.gammaln(alpha)) # ln[C(α)], refer to B.23 in PRML
    params['logWishartConst'] = np.zeros((K, 1))   # B(W, ν), refer to B.79 in Page 693, PRML
    for i in range(K):
        logdetW[i] = np.log(sl.det(W[i]))
        params['logLambdaTilde'][i] = np.sum(sp.digamma(0.5 * (v[i] + 1 - np.linspace(1, D, D)))) + \
            D * np.log(2) + logdetW[i]    # refer to equation 10.65 in Page 478, PRML
        logB = -(v[i]/2)*logdetW[i] - (v[i]*D/2)*np.log(2) - (D*(D-1)/4)*np.log(np.pi) - \
            np.sum(sp.gammaln(0.5*(v[i]+1 - np.linspace(1, D, D))))  # refer to B.79 in Page 693, PRML
        params['logWishartConst'][i] = logB
        params['entropy'][i] = -logB - (v[i] - D - 1) * params['logLambdaTilde'][i] / 2 + v[i] * D / 2

    return params

def parseParams(params):
    return params['alpha'], params['beta'], params['m'], params['v'], params['W'], params['logPiTilde'], \
           params['logLambdaTilde'], params['entropy'], params['logDirConst'], params['logWishartConst']

def lowerbound(priorParams, postParams, Nk, xbar, S, r):
    alpha, beta, m, v, W, logPiTilde, logLambdaTilde, entropy, logDirConst, logWishartConst = parseParams(postParams)
    alpha0, beta0, m0, v0, W0, logPiTilde0, logLambdaTilde0, entropy0, logDirConst0, logWishartConst0 = parseParams(priorParams)

    K, D = m.shape
    ElogpX = np.zeros(K)
    for i in range(K):
        xc = (xbar[i] - m[i]).reshape(1, -1)
        val = v[i] * np.sum(np.diag(np.dot(S[i], W[i])))
        val2 = v[i] * np.dot(xc, np.dot(W[i], xc.T))
        ElogpX[i] = Nk[i] * (logLambdaTilde[i] - D / beta[i] - val - val2 - D * np.log(2 * np.pi))
    ElogpX = 0.5 * np.sum(ElogpX)

    NK = Nk.reshape(-1, 1)
    ElogpZ = np.sum(NK * logPiTilde)

    Elogppi = logDirConst0 + np.sum((alpha0 - 1) * logPiTilde)

    ElogmuSigma = np.zeros(K)
    for i in range(K):
        mc = (m[i] - m0[i]).reshape(1, -1)
        ElogmuSigma[i] = 0.5 * (D * np.log(beta0[i] / (2 * np.pi)) + logLambdaTilde[i] - D * beta0[i] / beta[i] -\
            beta0[i] * v[i] * np.dot(mc, np.dot(W[i], mc.T))) + logWishartConst0[i] +\
            0.5 * (v0[i] - D - 1) * logLambdaTilde[i] - 0.5 * v[i] * np.sum(np.diag(np.dot(sl.inv(W0[i]), W[i])))
    ElogmuSigma = np.sum(ElogmuSigma)

    logr = np.log(r)
    ElogqZ = np.sum(np.sum(r * logr, axis=1))

    Elogqpi = np.sum((alpha - 1) * logPiTilde) + logDirConst

    ElogqmuSigma = np.sum(0.5 * logLambdaTilde + 0.5 * D * np.log(beta / (2 * np.pi)) - D/2 - entropy)

    L = ElogpX + ElogpZ + Elogppi + ElogmuSigma - ElogqZ - Elogqpi - ElogqmuSigma

    if np.isnan(L):
        raise ValueError('L should not be nan')

    return L

# calculate sufficient statistics
def computeESS(X, r):
    N, K = r.shape
    D = X.shape[1]
    Nk = np.sum(r, axis=0)    # 10.51
    Nk = Nk + 1e-10  # for numerical stable
    xbar = np.zeros((K, D))
    S = np.zeros((K, D, D))
    for i in range(K):
        rk = r[:, i].reshape(-1, 1)
        xbar[i] = np.sum(rk * X, axis=0) / Nk[i]      # 10.52
        xc = X - xbar[i]
        xc2 = rk * xc   # N * D
        xc2_3d = xc2.reshape(N, D, 1)
        xc_3d = xc.reshape(N, 1, D)
        S[i] = np.sum(xc2_3d * xc_3d, axis=0) / Nk[i] # 10.53

    return Nk, xbar, S

# update postParams
def MStep(Nk, xbar, S, priorParams):
    NK = Nk.reshape(-1, 1)
    alpha0 = priorParams['alpha']
    m0 = priorParams['m']
    beta0 = priorParams['beta']
    W0 = priorParams['W']
    v0 = priorParams['v']

    alpha = alpha0 + NK    # 10.58
    beta = beta0 + NK      # 10.60
    v = v0 + NK            # 10.63, maybe v0 + NK + 1
    m = np.zeros(m0.shape)
    W = np.zeros(W0.shape)
    for i in range(len(m)):
        if NK[i] < 0.001:  # extinguished
            m[i] = m0[i]
            W[i] = W0[i]
            v[i] = v0[i]
        else:
            m[i] = (beta0[i] * m0[i] + NK[i] * xbar[i]) / beta[i]
            gap = (xbar[i] - m0[i]).reshape(-1, 1)
            invW = sl.inv(W0[i]) + NK[i] * S[i] + (beta0[i] * NK[i] /(beta0[i] + NK[i])) * np.dot(gap, gap.T)
            W[i] = sl.inv(sl.inv(W0[i]) + NK[i] * S[i] + (beta0[i] * NK[i] /(beta0[i] + NK[i])) * np.dot(gap, gap.T))

    postParams = mixGaussBayesStructure(alpha, beta, m, v, W)
    return postParams

def EStep(postparams, X):
    m = postparams['m']
    beta = postparams['beta']
    W = postparams['W']
    v = postparams['v']
    logPiTilde = postparams['logPiTilde']
    logLambdaTilde = postparams['logLambdaTilde']
    K, D = m.shape
    N = X.shape[0]

    E = np.zeros((N, K))
    for i in range(K):
        XC = X - m[i].ravel()
        tmp = np.sum(np.dot(XC, W[i]) * XC, axis=1)  # refer to equation 10.64 in Page 478, PRML
        E[:, i] = D / beta[i] + v[i] * tmp

    logRho = (logPiTilde + 0.5 * logLambdaTilde).ravel() - 0.5 * E  # 10.46, ignore the costant D/2 * ln(2*pi)
    logSumRho = sp.logsumexp(logRho, axis=1)
    logr = logRho - logSumRho.reshape(-1, 1)
    r = np.exp(logr)
    Nk = np.exp(sp.logsumexp(logr, axis=0))
    r[r == 0] = 1e-40
    '''
    Rho = np.exp(logRho)
    r = Rho / np.sum(Rho, axis=1).reshape(-1, 1)
    Nk = np.sum(r, axis=0)
    '''

    return r, Nk

def VBEM(prior_init, posterior_init, X, maxIter):
    L_last = -np.inf
    prior, posterior = prior_init, posterior_init
    lowerbounds = []
    params = []
    for i in range(maxIter):
        r, Nk = EStep(posterior, X)
        Nk, xbar, S = computeESS(X, r)
        L = lowerbound(prior, posterior, Nk, xbar, S, r)
        lowerbounds.append(L)
        params.append(posterior)

        posterior_new = MStep(Nk, xbar, S, prior)
        posterior = posterior_new

        if np.allclose(L, L_last):
            print('Converged~')
            break

        L_last = L

    return params, lowerbounds