import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib.animation as ma

class BayesianSequential(object):
    def __init__(self, ax, proba=0.5):
        self.success = 0
        self.fail = 0
        self.proba = proba
        self.ax = ax

        # Set up plot parameters
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 15)
        self.ax.grid(True)

        # This vertical line represents the theoretical value, to
        # which the plotted distribution should converge.
        self.ax.axvline(proba, linestyle='--', color='black')

    def init(self):
        self.success = 0
        self.fail = 0
        return self.ax.plot([], [], 'k-')
        
    def beta_plot(self):
        '''
        根据alpha和beta画一个beta分布的图形，当alpha=beta=1时，是[0, 1]上的均匀分布
        '''
        x = np.linspace(0, 1, 200)
        beta = ss.beta(self.success, self.fail)
        y = beta.pdf(x)
        return self.ax.plot(x, y, color='black')
        # plt.show()

    def trial(self):
        '''
        模拟一次伯努利实验，proba指定成功的概率
        '''
        if np.random.randn(1,) < self.proba:
            return True;
        else:
            return False

    def __call__(self, i):
        if i == 0:
            return self.init()
        
        if self.trial():
            self.success += 1
        else:
            self.fail += 1

        # 计算后验的贝叶斯概率
        # print(self.success, self.fail)
        return self.beta_plot()

fig, ax = plt.subplots()
bs = BayesianSequential(ax, proba=0.7)
anim = ma.FuncAnimation(fig, bs, frames=np.arange(100), init_func=bs.init, interval=100, blit=True)
plt.show()
