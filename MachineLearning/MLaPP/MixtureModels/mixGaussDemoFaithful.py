import numpy as np
import scipy.io as sio
import scipy.stats as ss
import sklearn.preprocessing as sp
import matplotlib.pyplot as plt
import matplotlib.animation as ma

def GetColors(rmat):
    '''根据r矩阵来调配每个数据点的红色和蓝色墨水的比例'''
    colors = []
    for i in range(len(rmat)):
        ratio = rmat[i, 0] / (rmat[i, 0] + rmat[i, 1])  # 红色墨水的比例
        r = ratio
        g = 0
        b = 1 - ratio                          # 蓝色墨水的比例
        colors.append([r, g, b])

    return colors

def Draw(ax, data, r, mu, cov, singleColor=False):
    '''
    ax: 图形对象
    data: 数据点集，本例中都是不变的
    r: 两个cluster对每个数据点的责任矩阵，即E步中计算到的结果，反映在图形中是红色和蓝色的墨水比例
    mu: 两个cluster的均值向量
    cov: 两个cluster的协方差阵
    singleColor: 如果为True，则忽略掉r参数，只用绿色画出所有的点集
    '''
    rv1 = ss.multivariate_normal(mu[0], cov[0])  # 红色的
    rv2 = ss.multivariate_normal(mu[1], cov[1])  # 蓝色的
    X, Y = np.meshgrid(np.linspace(-2.4, 2.4, 200), np.linspace(-2.4, 2.4, 200))
    Z = np.dstack((X, Y))
    Z1 = rv1.pdf(Z)
    Z2 = rv2.pdf(Z)
    ratio = 0.6
    if singleColor:
        ratio = 0.9
    level_Z1 = Z1.min() + ratio * (Z1.max() - Z1.min())   # 只画一个圆，囊括 (1 - ratio) * 100% 的概率
    level_Z2 = Z2.min() + ratio * (Z2.max() - Z2.min())
    ax1 = ax.contour(X, Y, Z1, levels=[level_Z1], colors='red')
    ax2 = ax.contour(X, Y, Z2, levels=[level_Z2], colors='blue')
    ax3 = None
    artists = []
    if singleColor:
        ax3 = ax.scatter(data[:, 0], data[:, 1], c='green', s=20)
    else:
        mixColors = GetColors(r)
        ax3 = ax.scatter(data[:, 0], data[:, 1], c=mixColors, s=20)

    # 需要返回一个iterable的Artistax.plot返回的collection符合，但是scatter的PathCollection不符合，所以要加个逗号在后面
    artists.append(ax1.collections)
    artists.append(ax2.collections)
    artists.append(ax3)

    return artists

def GetResponsibility(x, pi, mu, cov):
    r = []
    total = 0
    for k in range(len(pi)):
        pi_k = pi[k]
        gaussianProb_k = ss.multivariate_normal(mu[k], cov[k]).pdf(x)
        rk = pi_k * gaussianProb_k
        
        r.append(rk)
        total += rk

    return r / total

def GetCovK(data, mu_k, rk):
    dim = len(mu_k.ravel())
    cov = np.zeros((dim, dim))
    for i in range(len(data)):
        rik = rk.ravel()[i]
        gap = data[i].reshape(-1, 1) - mu_k.reshape(-1, 1)
        cov_i = rik * np.dot(gap, gap.T)
        cov += cov_i

    return cov / np.sum(rk)
    

def EM(data, pi, mu, cov):
    # E step, 主要是计算r(ik)
    r = []
    for i in range(len(data)):
        xi = data[i]
        r.append(GetResponsibility(xi, pi, mu, cov))

    r = np.array(r)
    # M step, 重新计算pi, mu, cov
    pi_new = np.mean(r, axis=0)
    mu_new = []
    cov_new = []
    for k in range(len(mu)):
        rk = r[:, k].reshape(1, -1)
        mu_k = np.dot(rk, data) / np.sum(rk)
        mu_new.append(mu_k.ravel())
        cov_new.append(GetCovK(data, mu_k, rk))

    print('mu_new: \n', np.array(mu_new))
    print('cov_new: \n', np.array(cov_new))
    # print('r: \n', r)
    return r, pi_new, np.array(mu_new), np.array(cov_new)
        

class SeqUpdate(object):
    def __init__(self, ax, data):
        self.data = data
        self.x = np.linspace(-2.4, 2.4, 200)
        self.y = np.linspace(-2.4, 2.4, 200)

        self.ax = ax
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-2.4, 2.4)
        self.ax.set_ylim(-2.4, 2.4)
        self.ax.set_xticks([-2, 0, 2])
        self.ax.set_yticks([-2, 0, 2])

    def init(self):
        self.pi = np.array([0.5, 0.5])                     # 混合系数pi的初始化值
        self.mu = np.array([[-1.8, 1.6], [1.8, -1.6]])     # 初始化cluster的均值向量
        self.cov = np.array([np.eye(2), np.eye(2)])        # 初始化两个协方差矩阵

        return Draw(self.ax, self.data, None, self.mu, self.cov, True)

    def __call__(self, i):
        # 对于contour()这些不兼容于FuncAnimation的对象来说，需要额外调用以下ax.clear(),
        # 具体为什么，我也不知道，偶然试出来的
        self.ax.clear()  
        self.ax.set_title('Iteration: {0}, pi: {1}'.format(i + 1, self.pi.ravel()))
        self.r, self.pi, self.mu, self.cov = EM(self.data, self.pi, self.mu, self.cov)
        
        return Draw(self.ax, self.data, self.r, self.mu, self.cov, False)

# load data
faithful = sio.loadmat("faithful.mat")
# print('old faithful data: \n', faithful)
data = faithful['faithful']
print('data.shape: ', data.shape)
data = sp.StandardScaler().fit_transform(data)  # 标准化，有利于更快的收敛

# Animation
maxIteration = 20
fig, ax = plt.subplots()
fig.canvas.set_window_title('mixGaussDemoFaithful')

su = SeqUpdate(ax, data)
# 需要额外调用一下初始化函数..否则第一下不会展示出来，
# 用plot()这些都好好的，contour里面有bug，需要写很多额外代码来保证动画的演示效果
su.init()
# blit参数不要设置为True，会有一些诡异的bug
anim = ma.FuncAnimation(fig, su, frames=maxIteration, init_func=su.init, interval=500, repeat=False)

plt.show()

