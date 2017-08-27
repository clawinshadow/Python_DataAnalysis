import numpy as np
import scipy.linalg as sl
import scipy.optimize as so
import matplotlib.pyplot as plt

'''
y值服从拉普拉斯分布，不再是高斯分布，从而变得对离群点不敏感，更加的健壮
'''

# 生成带离群点的样本数据，10个正常样本，3个离群点[[0.1, -5], [0.5, -5], [0.9, -5]]
w_true = [0, 3]                              # y = 0 + 3*x
noise = 0.5 * np.random.randn(10)            # 噪声服从 N(0, 0.5 ** 2)

x = np.linspace(0, 1, 200)
randIndices = np.random.randint(0, 199, 10)         # 抽取10 个正常样本点 
x_sample = x[randIndices]               
y_sample = w_true[0] + w_true[1] * x_sample + noise # 带噪声的y值

# 3个离群点
x_outliers = [0.1, 0.5, 0.9]
y_outliers = np.tile(-5, 3)

x_sample = np.r_[x_sample, x_outliers]       # 带上离群点的样本
y_sample = np.r_[y_sample, y_outliers]

fig = plt.figure(figsize=(7, 6))
fig.canvas.set_window_title('linregRobustDemoCombined')

# plot sample points
plt.subplot()
plt.title('Linear data with noise and outliers', fontdict={ 'fontsize': 10 })
plt.axis([0, 1, -6, 4])
plt.xticks(np.arange(0, 1.1, 0.2))
plt.yticks(np.arange(-6, 4.1, 1))
plt.plot(x_sample, y_sample, ls='None', color='k', marker='o', fillstyle='none', markeredgewidth=2)

# 计算高斯分布假设下的权重向量, 直接用公式计算
dm = np.c_[np.ones(len(x_sample)).reshape(-1, 1), x_sample.reshape(-1, 1)]
y_vec = y_sample.reshape(-1, 1)
w_MLE = np.dot(sl.inv(np.dot(dm.T, dm)), np.dot(dm.T, y_vec))
print('w_MLE with Gaussian distribution: ', w_MLE.ravel())

# plot MLE
x_scatter = np.linspace(0, 1, 11)    # 等距离的11个样本点，为了画方格和circle
y_scatter_MLE = w_MLE[0] + w_MLE[1] * x_scatter
plt.plot(x_scatter, y_scatter_MLE, color='r', marker='o', \
         fillstyle='none', markeredgewidth=2, label='least squares')

# 计算拉普拉斯分布下的权重向量，用线性规划的算法来计算
# 有 D+2N 个未知参数，3N个约束条件..
# 本来是p(y|x,w, b) = Lap(y|w.T*x, b) ∝ exp(−1/b * |y − w.T*x|)
# 从而 NLL = Σ|ri(w)|，无法直接用最优化问题的算法求解
# 所以需要引入 r+ 和 r-, 转化成形如以下的线性规划问题：
# min Σ(r+.i + r-.i)  s.t. r+.i >= 0, r-.i >= 0, w.T * xi + r+.i - r-.i = yi
# 转化为矩阵的形式就是 min f.T*θ, s.t. Aθ <= b, A(eq)θ = b(eq), l <= θ <= u
# θ = (w, r+, r−), f = [0, 1, 1], A = [], b = [], Aeq = [X, I,−I], beq = y, l = [−inf, 0, 0]

# 使用scipy.optimize.linprog来计算，算法估计是单纯形下山法，收敛速度较慢
# 第一步，构造目标函数的系数矩阵，w占两个长度，r+和r_都是N个长度
c = np.r_[np.zeros(2), np.ones(len(x_sample)), np.ones(len(x_sample))]
# 因为本例中没有不等式约束(r+.i >= 0, r-.i >= 0放在后面的bounds里面)，所以省略
# 第二步，构造等式矩阵左边的系数矩阵A_eq，N*(D+2N)矩阵, 列数要与未知参数的数量一致，没有的用0补齐
A_EQ = []
for i in range(len(x_sample)):
    xi = x_sample[i]
    yi = y_sample[i]
    coef = np.zeros(len(c))     # 每一行的列数必须与未知参数c的长度一致，先全部填零
    coef[0:2] = [1, xi]         # 对应于前两个未知参数 w0, w1
    coef[2*i+2:2*i+4] = [1, -1] # 对应于r+.i, r-.i
    A_EQ.append(coef)

# 第三步，构造等式矩阵右边的b_eq
b_EQ = y_sample
# 第四步，构造bounds, 每个系数对应一个tuple，形如(None, None)，本例中w没有边界，r+和r-有
bounds = [(None, None), (None, None)]
for i in range(2 * len(x_sample)):
    bounds.append((0, None))

bounds = tuple(bounds)

# 使用so.linprog来计算
res = so.linprog(c, A_eq=A_EQ, b_eq=b_EQ, bounds=bounds)
print('Optimization Result by Linear Program: \n', res)
w_linprog = res.x[0:2]
print('Robust w by linprog: ', w_linprog)

# 使用scipy.optimizize.minimize() + huberLoss 来求解，huberLoss的好处就在于它是可导的
def loss(x, delta, x_train, y_train):
    # theta是权重参数，huberLoss的阈值delta还是要先定义好，这个可以用CV去求个比较好的值
    # 如果delta不事先定义好的话，最优化求解它会恒等于0，因为这时候loss的值是0，肯定最小
    residual = y_train - x[0] - x[1] * x_train
    print("residual min, max: ", residual.min(), residual.max())
    lossVals = []
    for i in range(len(residual)):
        if np.abs(residual[i]) <= delta:
            lossVals.append(0.5 * residual[i]**2)                                                   
        else:
            lossVals.append(delta * np.abs(residual[i]) - 0.5 * delta**2)

    return np.sum(lossVals)

delta = 1
x0 = [1, 1]
res_huber = so.minimize(loss, x0, args=(delta, x_sample, y_sample))
w_huberLoss = res.x
print('Robust w by so.minimize with huberLoss: ', w_huberLoss)
print('Optimization Result by so.minimize() with huberLoss delta={0}: \n{1}'.format(delta, res_huber))                             

# plot robust regression
y_scatter_robust = w_linprog[0] + w_linprog[1] * x_scatter
plt.plot(x_scatter, y_scatter_robust, color='blue', marker='s', ls=':',\
         fillstyle='none', markeredgewidth=2, label='laplace')
y_scatter_huber =  w_huberLoss[0] + w_huberLoss[1] * x_scatter
plt.plot(x_scatter, y_scatter_huber, color='cyan', marker='^', ls='-.',\
         fillstyle='none', markeredgewidth=2, label='laplace with huberLoss')

plt.legend()
plt.show()

