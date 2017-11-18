import numpy as np
import scipy.linalg as sl

'''
gradient descent method in essential
'''

def glmtrain(x, y, alpha, beta):
    '''
    实际上就是个岭回归的模型，只不过没有用lambda而是用的alpha和beta作为regularize的参数
    :param x:  assume the first column is 1,
    :param y:  target vector
    :param alpha: hyper-parameter of prior w, precision, scalar
    :param beta:  noise hyper-parameter, precision too, scalar
    :return: weights including intercept
    '''
    N, D = x.shape
    y = y.reshape(-1, 1)
    hessian = beta * np.dot(x.T, x) + alpha * np.eye(D)
    result = beta * np.dot(sl.inv(hessian), np.dot(x.T, y))

    return result

def rearrange_hess(nparams, nin, nout, j, out_hess, hdata):
    bb_start = nparams - nout
    ob_start = j * nin
    ob_end = (j + 1)  * nin
    b_index = bb_start + j

    hdata[ob_start:ob_end, ob_start:ob_end] = out_hess[:nin, :nin]
    hdata[b_index, b_index] = out_hess[nin, nin]
    hdata[b_index, ob_start:ob_end] = out_hess[nin, :nin]
    hdata[ob_start:ob_end, b_index] = out_hess[:nin, nin]

def hbayes(nparams, alpha, beta, hdata):
    h = beta * hdata
    h = h + alpha * np.eye(nparams)

    return h, hdata

def errbayes(alpha, beta, edata, w):
    e1 = beta * edata
    eprior = 0.5 * np.dot(w.T, w)
    e2 = eprior * alpha
    e = e1 + e2

    return e, edata, eprior

def glmerr(x, y, w, alpha, beta):
    y_estimate = np.dot(x, w)  # N * 1
    err = 0.5 * np.sum((y - y_estimate)**2)

    return errbayes(alpha, beta, err, w)

def glmhess(x, y, alpha, beta):
    N, D = x.shape
    nparams = D
    nin = D - 1
    nout = y.shape[1]
    out_hess = np.dot(x.T, x)  # D * D
    hdata = np.zeros((nparams, nparams))
    for j in range(nout):
        rearrange_hess(nparams, nin, nout, j, out_hess, hdata)

    return hbayes(nparams, alpha, beta, hdata)

def evidence(x, y, w, num, alpha, beta):
    ndata = len(x)
    h, dh = glmhess(x, y, alpha, beta)

    evl, vr = sl.eig(dh)
    evl = np.real(evl)
    evl[evl <= 0] = 1e-40
    evl = evl.reshape(-1, 1)

    e, edata, eprior = glmerr(x, y, w, alpha, beta)

    nout = y.shape[1]
    alpha_old, beta_old = alpha, beta
    for k in range(num):
        # update alpha
        L = beta_old * evl
        gamma = np.sum(L / (L + alpha_old))
        alpha_new = 0.5 * gamma / eprior
        logev = 0.5 * len(w) * np.log(alpha_new)

        # update beta
        beta_new = 0.5 * (nout * ndata - gamma) / edata
        logev += 0.5 * ndata * np.log(beta_new) - 0.5 * ndata * np.log(2 * np.pi)

        e, tmp1, tmp2 = errbayes(alpha_new, beta_new, edata, w)
        logev = logev - e - 0.5 * np.sum(np.log(beta_new * evl + alpha_new))

        alpha_old, beta_old = alpha_new, beta_new

    return logev, alpha_new, beta_new

def linregFitEB(x, y):
    alpha = 0.01
    beta = 0.05
    nouter = 5
    ninner = 2
    for k in range(nouter):
        w = glmtrain(x, y, alpha, beta)
        logev, alpha_new, beta_new = evidence(x, y, w, ninner, alpha, beta)
        alpha = alpha_new
        beta = beta_new

    return logev, alpha, beta, w
