import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

def Sample(N=200):
    rv1 = ss.norm(-10, 5)
    rv2 = ss.norm(10, 5)
    samples = []
    for i in range(N):
        pi = np.asscalar(np.random.rand(1))
        if pi > 0.5:
            samples.append(rv1.rvs(1))
        else:
            samples.append(rv2.rvs(1))

    return np.array(samples).ravel()

samples = Sample()
print('samples: ', samples)

# plots
fig = plt.figure(figsize=(11, 5))
fig.canvas.set_window_title('mixGaussLikSurfaceDemo')

plt.subplot(121)
plt.axis([-25, 25, 0, 35])
plt.xticks(np.arange(-25, 26, 5))
plt.yticks(np.arange(0, 36, 5))
plt.hist(samples, np.arange(-25, 26, 5), color='darkblue', edgecolor='k')

# calculate likelihood
def CalcLikelihood(samples, mu1, mu2):
    rv1 = ss.norm(mu1, 5)
    rv2 = ss.norm(mu2, 5)

    probs = 0.5 * rv1.pdf(samples) + 0.5 * rv2.pdf(samples)
    logSum = np.sum(np.log(probs)) # 计算log之和，避免下溢

    return logSum

def GetLiks(samples, mu1, mu2):
    result = []
    for i in range(len(mu1)):
        result.append(CalcLikelihood(samples, mu1[i], mu2[i]))

    return np.array(result).ravel()

x, y = np.meshgrid(np.linspace(-20, 20, 100), np.linspace(-20, 20, 100))
x_flat = x.ravel()
y_flat = y.ravel()
liks = GetLiks(samples, x_flat, y_flat)
z = liks.reshape(x.shape)

# plot likelihood surface
plt.subplot(122)
plt.axis([-20, 20, -20, 20])
plt.xticks(np.arange(-19.5, 19.6, 5))
plt.yticks(np.arange(-19.5, 19.6, 5))
plt.xlabel(r'$\mu_1$')
plt.ylabel(r'$\mu_2$')
plt.contour(x, y, z, cmap='jet')

plt.show()
        
    
