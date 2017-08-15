import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

'''
狄利克雷分布是分类分布中参数向量θ的共轭先验分布，也是一个K维的参数向量，记为α，生成出来的K维随机变量X:{x1, x2, ..., xk},
总和为1，很适合作为一个分类分布的参数，α：{α1, α2, ..., αk}控制着狄利克雷分布的各种性质，所有αi之和决定了狄利克雷分布的
尖峰到底有多尖，即它的众数有多么的集中，而各个α的值决定了分布的形状，如果是{0.1, 0.1, 0.1}，那么得出的样本会非常的稀疏，
即有很多个xi会是零，反映在图形中就是单纯形的各个角落都有一个尖峰，如果是{1, 1, 1}，则是均匀分布，如果是{2, 2, 2}，则是
众数为{1/3, 1/3, 1/3}的一个分布，但范围相对交广，不是很尖，如果是{20, 20, 20}，则概率分布紧紧的聚集在{1/3, 1/3, 1/3}周围，
会非常尖。
'''

alpha_sparse = np.tile(0.1, 5)
x_sparse = ss.dirichlet(alpha_sparse).rvs(5)
print("sparse x samples from dirichlet distribution: \n", x_sparse)

alpha_dense = np.tile(1, 5)
x_dense = ss.dirichlet(alpha_dense).rvs(5)

# plt.figure(figsize=(11, 6))
nrows, ncols = 5, 2
x = np.arange(1, 6, 1)

def Draw(samples, title):
    fig, ax = plt.subplots(5)
    fig.suptitle(title)
    fig.tight_layout()
    for i in range(5):
        ax[i].bar(x, samples[i], align='center', color='darkblue')
        ax[i].set_xlim(0, 6)
        ax[i].xaxis.set_ticks([0, 1, 2, 3, 4, 5])
        ax[i].yaxis.set_ticks([0, 0.5, 1])

Draw(x_sparse, 'Samples from Dir (alpha=0.1)')
Draw(x_dense, 'Samples from Dir (alpha=1)')
plt.show()
