import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

def getInterval(start, alpha, rv):
    '''计算cdf为alpha的区间'''
    start_cdf = rv.cdf(start)
    end_cdf = start_cdf + alpha
    end = rv.ppf(end_cdf)

    return end

def DrawWithInterval(index, alpha, CI):
    plt.subplot(index)
    plt.plot(x, y, 'k-', linewidth=2)
    plt.vlines(CI[0], 0, rv.pdf(CI[0]), color='darkblue', linewidth=2)
    plt.vlines(CI[1], 0, rv.pdf(CI[1]), color='darkblue', linewidth=2)
    plt.plot(CI, rv.pdf(CI), color='darkblue', linewidth=2)
    plt.xlim(0, 1)
    plt.ylim(0, 3.5)

rv = ss.beta(3, 9)
mode = (3 - 1) / ( 3 + 9 - 2) # beta分布的众数 (a + 1) / (a + b - 2)

x = np.linspace(0, 1, 1000)
y = rv.pdf(x)
alpha = 0.95               # 95%的置信度
CI = rv.interval(alpha)    # 置信区间，在贝叶斯里面是Credible Interval, 反正本质是一样的，名字不同而已

plt.figure(figsize=(11, 5))
DrawWithInterval(121, alpha, CI)

# 用蒙特卡罗方法来计算HPD区间，在0 和ppf(1-alpha)之间抽样，然后选最小的一个区间
max_ppf = rv.ppf(1 - alpha)
starts = np.linspace(0, max_ppf, 200)   # 区间的起始点
ends = getInterval(starts, alpha, rv)   # 区间的终点
ranges = ends - starts
sortedIndices = np.argsort(ranges)
min_interval = [starts[sortedIndices[0]], ends[sortedIndices[0]]]
print('HPD interval: ', min_interval)

DrawWithInterval(122, alpha, min_interval)

plt.show()

