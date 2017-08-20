import math
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib.animation as ma

'''
matplotlib里面有个功能很强大的模块叫Animation，很适合用于演示贝叶斯方法的序列化更新 Sequential Update
以后要经常用到，所以记一下FuncAnimation的各个参数的含义：
https://matplotlib.org/api/_as_gen/matplotlib.animation.FuncAnimation.html

它是一个class，建立好了后，直接调用plt.show()即可以开始演示
class matplotlib.animation.FuncAnimation(fig, func, frames=None, init_func=None, fargs=None, save_count=None, **kwargs)

Parameter:
    fig: Figure对象
    func: callable, 每次时间到了后需要调用的函数，第一个参数来自于frames(iterable,每次迭代都会更新的),
          后面的参数来自于fargs，每次迭代都共享的参数
          def func(fr: object, *fargs) -> iterable_of_artists:
    frames: 用于给func传参，生成每次迭代所需要更新的参数
          1. 如果是个迭代器，那么简单的传过去值就可以
          2. 如果是个整数，相当于range(frames)
          3. 如果是个生成函数，必须要是如下的签名：
             def gen_function() -> obj:
          4. 如果什么都没有，则默认传当前迭代的count， itertools.count
    init_func: 顾名思义，第一次迭代前要初始化的一些状态
    fargs: 每次调用func时都要传给他的参数，不变的
    save_count: frames要存到cache的数量，不太明白是干啥的
    interval: 每次更新的时间间隔，以毫秒为单位
    repeat_delay: 所有迭代都做完后，再重复下一次的时间间隔，毫秒为单位
    repeat: bool型，是否重复，默认为True
    blit: 一个性能参数，设置为True的话会快一点，默认是False

在一维的高斯分布中，inverse Wishart分布实际上就是inverse Gamma分布, 参数映射关系如下

    IW(σ2|s0, ν0) = IG(σ2| s0/2, v0/2)

对IG来说，mean = b / (a - 1) , mode = b / (a + 1)

# 特别注意 scipy.stats.invgamma 中的scale参数，与书里面惯用的b参数是对应起来的 scale = b
# invgamma.pdf(x, a, loc, scale) = invgamma.pdf(y, a) / scale , 而 y = (x - loc) / scale
# 再看pdf的计算公式:       invgamma.pdf(x, a) = x**(-a-1) / gamma(a) * exp(-1/x)
# 书里面IG(a,b)的计算公式：IG(a, b) = [b**a / Gamma(a)] * x**(-a-1) * exp(-b/x)
# loc可以不用管，通常都是0
'''

class SeqUpdate(object):
    '''
    __init__里面给一些恒定不变的图形属性和静态数据赋值
    init()里面是一些初始化的数据，比如最开始的先验分布的参数等
    __call__()是最核心的方法，每次迭代时要调用的方法，参数必有一个最新的样本
    '''
    def __init__(self, ax):
        self.x = np.linspace(0, 15, 1000)
        self.mu = 5

        self.ax = ax
        self.ax.grid(True)
        self.ax.axvline(10, linestyle='--', color='green')  # 理论上的方差值，10
        self.ax.set_xlim(0, 15)
        self.ax.set_ylim(0, 0.35)
        self.ax.set_title(r'$prior = IW(ν=0.001, S=0.001), true\ \sigma^2=10.000$')
        self.ax.set_xlabel(r'$\sigma^2$')
        
    def init(self):
        self.a0 = 0.0005    # 初始先验分布IG的a参数
        self.b0 = 0.0005    # 初始先验分布IG的b参数
        
        y = ss.invgamma.pdf(self.x, a=self.a0, scale=self.b0)
        return self.ax.plot(self.x, y, 'r-')  # 此处一定要return 这个artist
        
    def __call__(self, xi):
        self.a0 += 0.5                        # a0 = a0 + N/2, 此时N=1
        self.b0 += 0.5 * (xi - self.mu)**2    # b0 = b0 + 1/2 * sigma(xi-mu)**2
        y = ss.invgamma.pdf(self.x, a=self.a0, scale=self.b0)

        print(xi, self.a0, self.b0)        
        return self.ax.plot(self.x, y, 'r-')  # 此处一定要return这个artist，凡是回调函数都要return

fig, ax = plt.subplots()

# generate samples, 注意给定的是方差等于10，而norm里面要传标准差
samples = ss.norm(5, math.sqrt(10)).rvs(100)
su = SeqUpdate(ax)
anim = ma.FuncAnimation(fig, su, frames=samples, init_func=su.init, interval=200, repeat=False, blit=True)

plt.show()
