import numpy as np
import scipy.io as sio
import scipy.stats as ss
import scipy.linalg as sl
import matplotlib.pyplot as plt

'''
State Space Model, SSM, this demo is specially for LG-SSM, that is Linear-Gaussian SSM, or a LDS, 
Linear Dynamical System, paramters θt = (At, Bt, Ct, Dt, Qt, Rt), Bt and Dt used for the control signal ut,
it always be zeros, like in this demo. 

zt = At * z(t−1) + Bt * ut + εt, εt ∼ N(0,Qt)
yt = Ct * zt + Dt * ut + δt, δt ∼ N(0,Rt)

In kalman filter & kalman smoothing, parameters θt are all known, we just infer the hidden states zt
1. kalman filter: to calculate the below marginal posterior at time t
                  p(zt|y1:t, u1:t) = N (zt|μt, Σt)
                  
                  1. rt = yt - yt.estimate
                     yt.estimate = Ct * μ(t|t-1) + Dt * ut
                  2. St = Ct * Σ(t|t-1) * Ct.T + Rt
                  3. Kalman gain matrix: Kt = Σ(t|t-1) * Ct.T * sl.inv(St)
                  4. Σt = (I - Kt * Ct) * Σ(t|t-1)
                     μt = μ(t|t-1) + Kt * rt
                  
2. kalman smoothing: to calculate the below marginal posterior at time t, using future data.
                  p(zt |y1:T) = N (μ(t|T) , Σ(t|T) )
                  
                  1. forwards algorithm as the same in kalman filtering, to the end of graph, get p(zT|y1:T)
                  2. tracing backwards to get kalman smoothing result
                     
                     Jt = Σt * A(t+1).T * sl.inv(Σt+1), A(t+1) should be the same as At in this demo
                     Σ(t|T) = Σt + Jt * (Σt+1|T - Σ(t+1|t)) * Jt.T
                     μ(t|T) = μt + Jt * (μ(t|T+1) - μ(t+1|t))
                     
                  attention: 
                     μ(t+1|t) = At * μt + Bt * ut
                     Σ(t+1|t) = At * Σt * At.T + Q                     
                     
                  refer to Page.675
'''

def kalman_smoothing(y, A, C, Q, R, init_mu, init_V):
    init_mu = init_mu.reshape(-1, 1)
    K = len(init_mu)  # dimension of hidder state
    N, D = y.shape
    smooth_mus = np.zeros((N, K))
    smooth_covs = np.zeros((N, K, K))
    # forwards algorithm
    filter_mus, filter_covs = kalman_filter(y, A, C, Q, R, init_mu, init_V)

    smooth_mus[-1] = filter_mus[-1]
    smooth_covs[-1] = filter_covs[-1]
    # tracing backwards
    for i in range(N - 2, -1, -1):
        msmooth_future = smooth_mus[i + 1].reshape(-1, 1)
        Vsmooth_future = smooth_covs[i + 1]
        mfilt = filter_mus[i].reshape(-1, 1)          # μt
        Vfilt = filter_covs[i]                        # Σt
        mfilt_pre = np.dot(A, mfilt)                  # μ(t+1|t)
        Vfilt_pre = np.dot(A, np.dot(Vfilt, A.T)) + Q # Σ(t+1|t)

        J = np.dot(Vfilt, np.dot(A.T, sl.inv(Vfilt_pre)))
        smooth_mu = mfilt + np.dot(J, (msmooth_future - mfilt_pre))
        smooth_cov = Vfilt + np.dot(J, np.dot((Vsmooth_future - Vfilt_pre), J.T))

        smooth_mus[i] = smooth_mu.ravel()
        smooth_covs[i] = smooth_cov

    return smooth_mus, smooth_covs

def kalman_filter(y, A, C, Q, R, init_mu, init_V):
    '''
    ignore control signal B, D, u
    A should be a square matrix K * K, C should be a transform matrix D * K
    Q.shape = A.shape, R.shape: D * D
    '''

    init_mu = init_mu.reshape(-1, 1)
    K = len(init_mu)   # dimension of hidder state
    N, D = y.shape
    mus = np.zeros((N, K))
    covs = np.zeros((N, K, K))
    for i in range(N):
        if i == 0:
            pre_mu = init_mu
            pre_cov = init_V
            mpred = pre_mu
            vpred = pre_cov
        else:
            pre_mu = mus[i - 1].reshape(-1, 1)      # μ(t-1)
            pre_cov = covs[i - 1]                   # Σ(t-1)
            mpred = np.dot(A, pre_mu)                    # μ(t|t-1)
            vpred = np.dot(A, np.dot(pre_cov, A.T)) + Q  # Σ(t|t-1)

        yi = y[i].reshape(-1, 1)
        ye = np.dot(C, mpred)                       # D * 1
        r = yi - ye                                 # D * 1
        S = np.dot(C, np.dot(vpred, C.T)) + R       # D * D
        KM = np.dot(vpred, np.dot(C.T, sl.inv(S)))  # K * D
        new_mu = mpred + np.dot(KM, r)              # K * 1
        new_cov = np.dot(np.eye(K) - np.dot(KM, C), vpred)  # K * K

        mus[i] = new_mu.ravel()
        covs[i] = new_cov

    return mus, covs

# load data
data = sio.loadmat('kalmanTrackingDemo.mat')
print(data.keys())
x = data['x']
y = data['y']
print(x.shape, y.shape)
x = x.T
y = y.T
K = x.shape[1]
D = y.shape[1]
xx, yy = np.meshgrid(np.linspace(6, 23, 100), np.linspace(3, 15, 100))
xtest = np.c_[xx.ravel(), yy.ravel()]

# Fit with kalman filter & smoothing
F = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])
Q = 0.001 * np.eye(K)
R = np.eye(D)
initmu = np.array([8, 10, 1, 0])
initV = np.eye(K)

filter_mus, filter_covs = kalman_filter(y, F, H, Q, R, initmu, initV)
smooth_mus, smooth_covs = kalman_smoothing(y, F, H, Q, R, initmu, initV)
print(smooth_mus)

# plot sample data
fig = plt.figure(figsize=(13, 4))
fig.canvas.set_window_title('kalmanTrackingDemo')

ax1 = plt.subplot(131)
ax1.tick_params(direction='in')
plt.ylim(3, 15)
plt.xlim(6.4, 21.6)
plt.xticks(np.linspace(8, 20, 7))
plt.yticks(np.linspace(4, 14, 6))
plt.plot(y[:, 0], y[:, 1], 'go', mew=2, label='observed', linestyle='none', fillstyle='none')
plt.plot(x[:, 0], x[:, 1], 'ks', mew=2, label='truth', linestyle='-', lw=2, fillstyle='none')
plt.legend()

# plot Kalman filtering  &  smoothing result
def plot(index, label, mus, covs):
    ax2 = plt.subplot(index)
    ax2.tick_params(direction='in')
    plt.axis([6.4, 22.6, 3, 15])
    plt.xticks(np.linspace(8, 22, 8))
    plt.yticks(np.linspace(4, 14, 6))
    plt.plot(y[:, 0], y[:, 1], 'go', mew=2, label='observed', linestyle='none', fillstyle='none')
    plt.plot(mus[:, 0], mus[:, 1], 'rx-', lw=2, ms=8, label=label)
    for i in range(len(mus)):
        mu = mus[i][:2]
        cov = 0.1 * covs[i][:2, :2]
        zz = ss.multivariate_normal(mu, cov).pdf(xtest)
        zz = zz.reshape(xx.shape)
        level = zz.min() + 0.05 * (zz.max() - zz.min())
        plt.contour(xx, yy, zz, colors='blue', levels=[level])
    plt.legend()

plot(132, 'filtered', filter_mus, filter_covs)
plot(133, 'smoothed', smooth_mus, smooth_covs)

plt.tight_layout()
plt.show()