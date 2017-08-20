import numpy as np
import scipy.stats as ss
import scipy.linalg as sl
import matplotlib.pyplot as plt
import matplotlib.patches as mp

'''
Wishart 分布， 一般是作为多元高斯分布理面协方差矩阵参数的先验分布，由两个参数组成 Σ ~ Wi(Σ|S, ν).
ν是自由度，S是规模矩阵(scale matrix)，Wishart分布和高斯分布之间有个简单的联系，如果 xi~N(0, Σ), 则 Σxi*xi.T 服从一个Wishart分布
Wi(Σ, 1). 一般来说Wishart的均值和众数为：

    mean = vS, mode = (v - D - 1)S

D是随机变量的维度，只有当 v > D + 1 时众数才存在
如果 D = 1， Wishart分布降为Gamma分布Ga(λ|ν/2, s/2)

相应的，如果协方差的先验分布是Wishart(Σ|S, ν), 则它的逆矩阵 服从Inverse Wishart分布，记为 IW(S.inv, v + D + 1)
'''

def DrawCovByEllipse(fig, index, cov):
    u, v = sl.eigh(cov)                  # eigh()用于求解复埃尔米特矩阵或者实对称矩阵，比一般普通矩阵要快一点
    angle = np.arctan2(v[0][1], v[0][0]) # arctan2的参数有两个，坐标系四个区域的角度都有，得出来是个浮点数
    angle = (180 * angle / np.pi)        # 转化为适宜画图的角度，[-180, 180]的值域
    u_percent95 = 5 * (u**0.5)           # 此处乘以5，表示画出来的椭圆将囊括95%的概率密度，具体公式可去网上查
    center = [0, 0]                      # 本例中都以原点为中心点
    e = mp.Ellipse(center, u_percent95[0], u_percent95[1], angle)

    print('cov: \n', cov)
    print('angle is: ', angle)
    print('u_percent95: ', u_percent95)
    print('v: \n', v)
    
    ax = fig.add_subplot(index)
    ax.add_artist(e)
    ax.set_xlim(-1 * u_percent95[0], u_percent95[0])
    ax.set_ylim(-1 * u_percent95[1], u_percent95[1])
    e.set_clip_box(ax.bbox)
    e.set_facecolor('none')
    e.set_edgecolor('darkblue')

df = 3.0 # 自由度
S = np.array([[3.1653, -0.0262], [-0.0262, 0.6477]])

wishart = ss.wishart(df, S)
samples = wishart.rvs(9, random_state=3)
print('samples.shape: ', samples.shape)

fig = plt.figure(figsize=(9, 8))
plt.suptitle('Wi(dof=3.0, S), E=[9.5, −0.1; −0.1, 1.9], ρ=−0.0')

for i in range(len(samples)):
    index = (int)('33' + str(i + 1))
    DrawCovByEllipse(fig, index, samples[i])
    
plt.show()
