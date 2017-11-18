import numpy as np
import scipy.linalg as sl
import scipy.special as sp

'''
refer to Page 747. 21.5.2

% The underlying generative model assumes
%
% p(y | x, w, lambda) = N(y | w'x, lambda^-1),
%
% with x and y being the rows of the given X and y. w and tau are assigned
% the conjugate normal inverse-gamma prior
%
% p(w, lambda | alpha) = N(w | 0, (lambda alpha)^-1 I) Gam(lambda | a0, b0),
%
% with the hyper-prior
%
% p(alpha) = p(alpha | c0, d0).
与书里面不一样的是这里alpha的先验分布用c0和d0来表示

prior : 
p(w, λ, α) = N (w|0, (λα)−1 * I)Ga(λ|a0 , b0 )Ga(α|c0 , d0 )

approx:
q(w, α, λ) = q(w, λ)q(α)

vb result:
q(w, α, λ ) = N(w|wN , λ^-1 * VN)*Ga(λ|aN, bN )*Ga(α|cN, dN)

vb的evidence没法计算，只能用lower bound代替
'''

# slightly different with matlab code, because data precision difference, every tmp data in matlab code is .4f
# so there is a cumulative error in the result
def linregFitVB(x, y):
    # uninformative priors
    a0 = 1e-6
    b0 = 1e-6
    c0 = 1e-6
    d0 = 1e-6

    N, D = x.shape
    x_cov = np.dot(x.T, x)
    xy = np.dot(x.T, y)
    aN = a0 + N / 2       # 在整个迭代过程中aN都不会再改变
    cN = c0 + D / 2       # 同上
    gammaln_aN = sp.gammaln(aN)
    gammaln_cN = sp.gammaln(cN)

    L_last = -np.inf
    maxIter = 500
    E_a = c0 / d0
    for i in range(maxIter):
        invV = E_a * np.eye(D) + x_cov
        V = sl.inv(invV)
        logdetV = -np.log(sl.det(invV))
        w = np.dot(V, xy)

        sse = np.sum((np.dot(x, w) - y)**2)
        bN = b0 + 0.5 * (sse + E_a * np.dot(w.T, w))
        E_t = aN / bN

        dN = d0 + 0.5 * (E_t * np.dot(w.T, w) + np.sum(np.diag(V)))
        E_a = cN / dN

        L = - 0.5 * (E_t * sse + np.sum(np.dot(x.T, np.dot(x, V)))) + 0.5 * logdetV \
            - b0 * E_t + gammaln_aN - aN * np.log(bN) + aN + gammaln_cN - cN * np.log(dN)

        if L < L_last:
            print('Last bound {0:.5f} greater than current bound {1:.5f}'.format(L_last, L))
            raise ValueError('Variational bound should not reduce')

        if np.allclose(L, L_last):
            print('Converged~')
            break

        L_last = L

    # add prior
    L = L - 0.5 * (N * np.log(2 * np.pi) - D) - sp.gammaln(a0) + a0 * np.log(b0) - sp.gammaln(c0) + c0 * np.log(d0);

    return w, V, aN, bN, cN, dN, L