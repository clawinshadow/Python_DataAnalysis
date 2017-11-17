import numpy as np
import scipy.io as sio
import scipy.stats as ss
import matplotlib.pyplot as plt

'''
这里需要特别注意的一点就是关于一元高斯分布的先验分布的不同表示方法，书里面出现了两种，很容易造成混淆

1. Page 136, 4.6.3.7, 这个先验分布采用的是方差，从而概率分布是一个NIX分布，即 Normal Inverse Chi-Squared
   具体如下：
   
   NIχ2(μ,σ2|m0,κ0,ν0,σ0**2) =>  N(μ|m0,σ2/κ0) * χ−2(σ2|ν0,σ0**2)
   
   所以这个先验分布的形式是 p(μ, σ2), 第二个参数是方差
   
2. Page 773， 21.5.1 这个先验分布采用的是precision，从而概率分布是一个Normal Gamma分布，
   具体如下：
   
   p(μ,λ) = N(μ|μ0,1／(κ0λ))Ga(λ|a0, b0)
   
   所以先验分布的形式是 p(μ, λ)，因为将第二个参数换成了precision，所以得到的概率分布也截然不同
   
两者之间的转换关系：

方差 σ2 ～ χ−2(σ2|ν0,σ0**2)， 这实际上是一个scaled inverse chi-squared 分布
它对应一个inverse gamma 分布，具体可参见wikipedia：https://en.wikipedia.org/wiki/Scaled_inverse_chi-squared_distribution

即：χ−2(σ2|ν0, σ0**2) = inv-gamma(ν0/2, ν0 * σ0**2 / 2)

所以：方差 σ2 ～ inv-gamma(ν0/2, ν0 * σ0**2 / 2)
又因为 λ = 1／σ2
所以 λ ～ Gamma(ν0/2, ν0 * σ0**2 / 2), a0 = ν0/2, b0 = ν0 * σ0**2 / 2
'''

def vb_pdf(mu_, lambda_, muN, kappaN, aN, bN):
    gaussian = ss.norm(muN, np.sqrt(1/kappaN))

    return gaussian.pdf(mu_) * ss.gamma.pdf(lambda_, aN, loc=0, scale=1/bN)

def normalGammaPdf(mu_, lambda_, mu0, kappa0, a0, b0):
    probs = np.zeros(mu_.shape)
    '''  # will be too slow
    for i in range(len(mu_)):
        mu = mu_[i]
        precision = lambda_[i]
        sigma = np.sqrt(1 / (kappa0 * precision))

        gauss_prob = ss.norm.pdf(mu, loc=mu0, scale=sigma)
        gamma_prob = ss.gamma.pdf(precision, a0, loc=0, scale=1/b0)  # 注意这个的scale要用倒数，可参考scipy中的文档
        probs[i] = gauss_prob * gamma_prob
    '''

    sigmas = np.sqrt(1 / (kappa0 * lambda_))
    gauss_prob = ss.norm.pdf(mu_, loc=mu0, scale=sigmas)  # scale参数可以支持数组，只是长度要与x一致
    gamma_prob = ss.gamma.pdf(lambda_, a0, loc=0, scale=1 / b0)  # 注意这个的scale要用倒数，可参考scipy中的文档
    probs = gauss_prob * gamma_prob

    return probs

# load data
data = sio.loadmat('unigaussVbDemo.mat')
print(data.keys())
x = data['data']
print(x)
m = np.mean(x)
N, D = x.shape

# prior hyper-parameters
mu0 = 0
kappa0 = 0
a0 = 0
b0 = 0

# true posterior, closed solution
muN_true = m
kappaN_true = N
aN_true = N / 2
bN_true = np.sum((x - m)**2) / 2

# initialize guess for VB
aN = 2.5
bN = 1
muN = 0.5
kappaN = 5

# data for plot
mus, lambdas = np.meshgrid(np.linspace(-1, 1, 200), np.linspace(0.0001, 2, 200))
z_true = normalGammaPdf(mus.ravel(), lambdas.ravel(), muN_true, kappaN_true, aN_true, bN_true)
z_true = z_true.reshape(mus.shape)
z_vb = vb_pdf(mus.ravel(), lambdas.ravel(), muN, kappaN, aN, bN)
z_vb = z_vb.reshape(mus.shape)

# iteration to fit VB
maxIter = 5
vb_probs = []
for i in range(maxIter):
    # update q(μ)
    muN = (kappa0 * mu0 + N * m) / (kappa0 + N)  # 每次迭代都一直是根据先验分布mu0, kappa0, ...来计算的
    kappaN = (kappa0 + N) * aN / bN              # 然后在每次迭代过程中轮流更新四个后验的参数
    if (i == 0):
        probs = vb_pdf(mus.ravel(), lambdas.ravel(), muN, kappaN, aN, bN)
        vb_probs.append(probs.reshape(mus.shape))

    # update q(λ)
    aN = a0 + (N + 1) / 2
    emu = muN
    emu2 = muN ** 2 + 1 / kappaN
    val1 = kappa0 * (emu2 + mu0 ** 2 - 2 * mu0 * emu)
    val2 = np.sum(x ** 2 + emu2 - 2 * emu * x)
    bN = b0 + val1 + 0.5 * val2
    if (i == 0):
        probs2 = vb_pdf(mus.ravel(), lambdas.ravel(), muN, kappaN, aN, bN)
        vb_probs.append(probs2.reshape(mus.shape))

probs3 = vb_pdf(mus.ravel(), lambdas.ravel(), muN, kappaN, aN, bN)
vb_probs.append(probs3.reshape(mus.shape))  # final result


# plot
fig = plt.figure(figsize=(10, 9))
fig.canvas.set_window_title('unigaussVbDemo')

def plot(index, probs_vb):
    ax1 = plt.subplot(index, aspect='equal')
    ax1.tick_params(direction='in')
    plt.axis([-1.1, 1.4, 0, 2])
    plt.xticks(np.linspace(-1, 1, 5))
    plt.yticks(np.linspace(0, 2, 11))
    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$\lambda$')
    cs = plt.contour(mus, lambdas, z_true, colors='g', linewidths=1)
    cs2 = plt.contour(mus, lambdas, probs_vb, colors='r', linewidths=1)
    artists, labels = cs.legend_elements()
    artists2, labels2 = cs2.legend_elements()
    plt.legend((artists[0], artists2[0]), ('exact', 'vb'))  # to simulate a legend for contour plot

plot(221, z_vb)
plot(222, vb_probs[0])
plot(223, vb_probs[1])
plot(224, vb_probs[2])

plt.tight_layout()
plt.show()