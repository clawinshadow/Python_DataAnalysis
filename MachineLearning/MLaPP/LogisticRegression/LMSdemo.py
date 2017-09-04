import numpy as np
import scipy.linalg as sl
import scipy.optimize as so
import sklearn.linear_model as slm
import matplotlib.pyplot as plt

''' 未完成 '''

'''
a kind of SGD, used for linear regression

SGD的算法中，比较重要的是步长的选择，Robins-Monro 条件给出了确保收敛性的eta必须满足的条件：

    Σ(ηk) = ∞, Σ(ηk**2) < ∞.

实际应用中有很多种公式可以用来选择步长，但用的比较多的是下面这个：

    ηk = (τ0 + k) ** −κ
    where : τ0 ≥ 0, κ ∈ (0.5, 1]

本例中采用 τ0 = 1, k = 1, 即步长为迭代次数的倒数
'''
def randRange(vmin, vmax, size=20):
    return vmin + (vmax - vmin) * np.random.rand(20)

# generate data
np.random.seed(0)
w_true = [1, 1]                   # 真实的权重向量
x = randRange(-4, 4, 20)
y = w_true[0] + w_true[1] * x + 0.8 * np.random.randn(20)    # N(0, 0.8**2)的噪声

def SSE(w, x_train, y_train):
    return np.sum(np.power(y_train - w[0] - w[1] * x_train, 2))

def GetEta(eta, X, y, w0, g):
    w = w0 - eta * g
    return SSE(w, X, y)

dm = np.c_[np.ones(len(x)).reshape(-1, 1), x.reshape(-1, 1)]
w_MLE = np.dot(sl.inv(np.dot(dm.T, dm)), np.dot(dm.T, y.reshape(-1, 1)))  # MLE

# initial values
eta0 = 0.1         # 0.1 太小，后面的步长随着k的增大会变得非常小。。w每次更新基本没什么变化，收敛的很慢
w0 = [-0.5, 2]
nupdates = 0
maxEpoches = 1000    # 书中只迭代了26次就有比较不错的结果了。。因为不知道它采用的哪种步长更新的逻辑，所以无法精确地实现

# Standard SGD, B = 1, eta = 1/k 
eta = eta0
w = w0
w_trace = [w0]
RSS_trace = [SSE]
dataset = np.c_[dm, y.reshape(-1, 1)]
for i in range(maxEpoches):
    np.random.shuffle(dataset)   # 每个epoch都要打乱数据集的顺序
    x_train = dataset[:, :2]
    y_train = dataset[:, -1]
    for j in range(len(x_train)):
        xk, yk = x_train[j], y_train[j]
        y_estimate = np.sum(w * xk)
        w_next = w - eta * (y_estimate - yk) * xk            # update w
        
        if sl.norm(w_next - w) < 1e-12:
            break
        
        eta = eta0 * (1 / (1 + nupdates))                    # update eta
        nupdates += 1               
        w = w_next
    w_trace.append(w)
    RSS_trace.append(SSE(w, x, y))

w_trace = np.array(w_trace)
RSS_trace = np.array(RSS_trace)

# use sklearn.SDGRegressor, 作为一个参考，因为是个随机算法，所以得出来的结果是有出入的
SGD = slm.SGDRegressor(penalty='none', max_iter=1000)
SGD.fit(x.reshape(-1, 1), y)
print('w by sklearn.SGDRegressor: ', SGD.intercept_, SGD.coef_)
print('n_iter: ', SGD.n_iter_)

print('w_trace: \n', w_trace)
print('n_updates: ', nupdates)
print('w_MLE: ', w_MLE)
print('RSS_trace: ', RSS_trace)

# plots
fig = plt.figure(figsize=(13, 6))
fig.canvas.set_window_title('LMSdemo')

plt.subplot(121)
plt.axis([-1, 3, -1, 3])
plt.xticks(np.arange(-1, 3.1, 1))
plt.yticks(np.arange(-1, 3.1, 0.5))
plt.xlabel('w0')
plt.ylabel('w1')
plt.title('black line = LMS trajectory towards LS soln (red cross)', fontdict={ 'fontsize': 10 })
plt.plot([w_MLE[0]], [w_MLE[1]], 'rx', markersize=10, markeredgewidth=2)

X, Y = np.meshgrid(np.linspace(-1, 3, 200), np.linspace(-1, 3, 200))
X_flat = X.ravel()
Y_flat = Y.ravel()
Z_flat = []
for i in range(len(X_flat)):
    Z_flat.append(SSE([X_flat[i], Y_flat[i]], x, y))
Z = np.array(Z_flat).reshape(X.shape)

plt.contour(X, Y, Z, cmap='jet')
plt.plot(w_trace[:, 0], w_trace[:, 1], 'k-')

plt.show()

        
    
    
