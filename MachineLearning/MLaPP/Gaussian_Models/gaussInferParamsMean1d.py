import math
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

'''
liklihood的参数是我根据图形猜的，书里面没写
在linear gaussian system中，隐藏变量x的分布p(x)是高斯分布，似然函数p(y|x)的分布也是高斯分布，这两个都要是已知的
这样我们才能得到公式化的解 p(y) 和 p(x|y)
相反，在4.3中我们是已知联合概率分布p(x, y)是一个高斯分布，去得到公式化的解 p(x), p(y) 和 p(x|y) 以及 p(y|x)
''' 

def Draw(index, var):
    plt.subplot(index)
    prior = ss.norm(0, math.sqrt(var))
    likelihood = ss.norm(3, 1)    # 这个是我估计的，均值大概是3，标准差是1
    posterior_var = (var * 1) / (1 * var + 1)
    posterior_mu = 3 * 1 * var / (1 * var + 1)
    posterior = ss.norm(posterior_mu, math.sqrt(posterior_var))
    
    x = np.arange(-5, 5, 0.01)
    print(prior.pdf(x))
    plt.plot(x, prior.pdf(x), 'b-')
    plt.plot(x, likelihood.pdf(x), 'r:')
    plt.plot(x, posterior.pdf(x), 'k-.')
    plt.xticks([-5, 0, 5])
    plt.yticks(np.linspace(0, 0.8, 0.1))
    plt.title("prior variance = 1.00")
    plt.xlim(-5, 5)
    plt.ylim(0, 0.7)
    

plt.figure(figsize=(11, 5))
Draw(121, 1)
Draw(122, 5)

plt.show()
