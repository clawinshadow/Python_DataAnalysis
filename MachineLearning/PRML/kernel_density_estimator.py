import numpy as np
import numpy.linalg as nl
import scipy.stats as ss
import matplotlib.pyplot as plt

'''
一种无参数的概率模型估计方法，需要使用所有的数据点，计算量随dataset的size而线性增长
'''

def Mixture_Gaussian(datas):
    '''
    制造一个混合的高斯分布模型，用于产生数据
    先生成独立的高斯分布，然后根据那个公式将每个pdf乘以一个系数，来构造混合的高斯分布。
    参数datas只能用于画图，它不是混合高斯模型出产的典型数据。
    最后生成一组用于kde的观测数据点，就是简单的把所有独立高斯分布产出的数据组合在一起就好了
    '''
    guassian_1 = ss.norm(-1, 1)         # 第一个高斯分布，mu = 0, sigma = 1
    guassian_2 = ss.norm(3, 0.5)        # 第二个高斯分布，mu = 3, sigma = 0.5

    data_1 = guassian_1.rvs(size=40)    # 这个size的比例一定要符合构造混合高斯分布的各个系数
    data_2 = guassian_2.rvs(size=60)    # 40 / 100, 60 / 100
    bimodal_data = np.concatenate([data_1, data_2]) # 组合成所有的观测数据点，这就是混合高斯模型的现实意义
    
    return bimodal_data, 0.4 * guassian_1.pdf(datas) + 0.6 * guassian_2.pdf(datas)

def Kernel_Density_Estimation(x, observations, h):
    '''
    采用高斯分布作为核函数，N个observations就是N个高斯分布相加, h是超方形的边长, x是空间中任意一点
    '''
    x = np.array(x)
    obs = np.array(observations)
    D = 0           # 计算维度
    if x.ndim == 0:
        D = 1
    elif x.ndim == 1:
        D = len(x)
    else:
        raise ValueError('x must be a scalar or 1-dimension array')
    N = 0
    if obs.ndim == 1:
        N = len(obs)
    else:
        N = obs.shape[1]

    kernel_probs = []
    for i in range(N):
        kernel_probs.append(np.exp(-1 * nl.norm(x - obs[i])**2 / (2 * h**2)) / np.power(2 * np.pi * h**2, D/2))

    return np.sum(kernel_probs) / N
    

'''
def Mixture_Gaussian2():
    ''''''
    这种方式是先根据独立的高斯分布来生成各自的观测数据点，然后将每组观测数据合并在一起，再根据
    每组的个数与总个数的占比来作为系数，最后计算出混合高斯分布的pdf。
    相比上个方法，这个不需要提前准备数据，只要确定每个独立的高斯分布，就可以返回一组数据点，并且
    包含每个数据点对应的pdf
    ''''''
    guassian_1 = ss.norm(-1, 1)           # 第一个高斯分布，mu = 0, sigma = 1
    guassian_2 = ss.norm(3, 0.5)          # 第二个高斯分布，mu = 3, sigma = 0.5

    data_1 = guassian_1.rvs(size=400)
    data_2 = guassian_2.rvs(size=600)
    pdf_1 = guassian_1.pdf(data_1)
    pdf_2 = guassian_2.pdf(data_2)
    
    #pdf = np.concatenate(pdf_1 *
    return data, 0.4 * guassian_1.pdf(datas) + 0.6 * guassian_2.pdf(datas)
'''

datas = np.linspace(-5, 5, 1000)            # 生成数据样本，1000个用来保证图线的光滑
observations, pdf = Mixture_Gaussian(datas)

kde_fit1, kde_fit2, kde_fit3 = [], [], []
h = [0.02, 0.4, 2]
for i in range(len(datas)):
    kde_fit1.append(Kernel_Density_Estimation(datas[i], observations, h[0]))
    kde_fit2.append(Kernel_Density_Estimation(datas[i], observations, h[1]))
    kde_fit3.append(Kernel_Density_Estimation(datas[i], observations, h[2]))

plt.figure(figsize=(11, 10))
ax = plt.subplot(411)                   # h = 0.02, 子区间太小，overfit
plt.plot(datas, pdf, color='green', ls='--', label='Mixture_Gaussian')
plt.plot(datas, kde_fit1, color='navy', label='KDE')
plt.text(0.45, 0.9, 'h = 0.02', color='firebrick', fontsize=12, transform=ax.transAxes)
plt.legend()

ax2 = plt.subplot(412)                   # h = 0.4, best fit
plt.plot(datas, pdf, color='green', ls='--')
plt.plot(datas, kde_fit2, color='navy', lw=1)
plt.text(0.45, 0.9, 'h = 0.4', color='firebrick', fontsize=12, transform=ax2.transAxes)

ax3 = plt.subplot(413)                   # h = 2, 子区间太大，失去了波动性
plt.plot(datas, pdf, color='green', ls='--')
plt.plot(datas, kde_fit3, color='navy', lw=1)
plt.text(0.45, 0.9, 'h = 2', color='firebrick', fontsize=12, transform=ax3.transAxes)

ax4 = plt.subplot(414)                              # h = 0.4, 使用stats提供的gaussian_kde方法来计算
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html#scipy.stats.gaussian_kde
kde = ss.gaussian_kde(observations, bw_method=h[0]) 
kde2 = ss.gaussian_kde(observations, bw_method='scott')
kde3 = ss.gaussian_kde(observations, bw_method='silverman')
plt.plot(datas, pdf, color='green', ls='--')
plt.plot(datas, kde(datas), color='navy', label='h = 0.02')   # 比我自己算的要更平滑，还没看scipy具体的代码，不清楚原因
plt.plot(datas, kde2(datas), color='pink', label='Scott\'s Rule')
plt.plot(datas, kde3(datas), color='yellow', label='Silverman''s Rule')
plt.text(0.4, 0.9, 'scipy.stats.gaussian_kde', color='firebrick', fontsize=12, transform=ax4.transAxes)
plt.legend()

plt.show()
