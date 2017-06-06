import numpy as np
import scipy.linalg as sl
import sklearn.linear_model as slm
import matplotlib.pyplot as plt

'''
Iterative Reweighted Least Square, 用于广义线性模型中对模型参数的最大似然估计，一般是用来求解线性模型中的
权重向量 w, 因为不是每个模型都想简单线性回归那样有通项公式的解，所以对于logistic回归这样的模型来说，就得
使用这种迭代的数值算法来寻找最优解，基础是Newton-Raphson的局部最优解求解理论，但因为cross entropy error
function E(w)是一个凸函数，所以这个局部最优解其实就是全局最优解，其中要用到Hessian矩阵

w.new = w.old - H.inv * D[E(w.old)].T, 用二阶导数矩阵的逆 乘以 E(W)的梯度矩阵来更新权重向量

在线性回归中，对应普通最小二乘法，因为此时：
误差函数：    E(w)      = **2, w是M * 1的权重向量，xn是第n个观测值向量，M * 1
梯度矩阵：    D[E(w)].T = 2*Σ(w.T * Φ(xn) - tn)*Φ(xn) = Φ.T*Φ*w - Φ.T*t, t是目标分类向量， N * 1
Hessian矩阵： H[E(w)]   = Φ.T*Φ
代入公式：    w.new = w.old - (Φ.T*Φ).inv * (Φ.T*Φ*w.old - Φ.T*Φ)
                    = (Φ.T*Φ).inv * Φ.T * t
正好消去了w.old, 直接得到了最终解析解，一步就收敛到位

在logistic回归中，yn = P(C1|xn), yn = σ(an), an = w.T * Φ(xn), 有：
D[E(w)].T = Σ(yn - tn)Φ(xn) = Φ.T(y - t), y和t分别是用旧权重计算出来的概率向量和对应的分类向量
H = Σyn(1 - yn)Φ(xn)*Φ(xn).T = Φ.T*R*Φ, 就比上面的普通最小二乘中间多了个N*N矩阵R
    R是一个对角矩阵，Rnn = yn (1 - yn), 其余的元素均为零

w.new = w.old - (Φ.T*R*Φ).inv * Φ.T * (y - t)
'''

def reweight(w_old, x, t):
    '''
    w_old: 旧的权重向量，M * 1
    x    : 经过基函数转化之后的样本矩阵，即design matrix，N * M
    t    : 目标分类向量，N * 1
    '''

    a = np.dot(x, w_old)                    # 将每一行观测值进行线性组合，得到一个标量值，a是N * 1
    y = 1 / (1 + np.exp(-a))                # 代入logistic函数，计算每一行对应的分类概率
    diag_vals = y * (1 - y)                 # 计算R矩阵对角线上的值
    r = np.diag(diag_vals.ravel())          # R矩阵，N * N
    # print(y, r)
    f1 = sl.inv(np.dot(np.dot(np.transpose(x), r), x))  # (Φ.T*R*Φ).inv
    f2 = np.dot(np.transpose(x), (y - t))               # Φ.T * (y - t)
    return w_old - np.dot(f1, f2)

def irls(x, t, tol=0.0001, iterLimit=200):
    x = np.c_[np.ones(t.shape[0]).reshape(-1, 1), x]  # 插入到第一列，全部为1，用来乘以截距w0
    w_old = np.zeros(x.shape[1]).reshape(-1, 1)       # 初始化权重向量，全部设为0
    iter_count = 0
    while True:
        iter_count += 1
        w_new = reweight(w_old, x, t)
        print('Iter Count {0}: {1}'.format(iter_count, w_new.ravel()))
        gap = np.absolute(w_new - w_old)
        if np.alltrue(gap < tol) or iter_count > iterLimit:
            return w_new
        else:
            w_old = w_new

def plot(w, x, t):
    x = x.ravel()
    t = t.ravel()
    w = w.ravel()
    # 生成sigmoid曲线的样本点
    x2 = np.linspace(-0.3, 6, 500)
    probs = 1 / (1 + np.exp(-(w[0] + w[1] * x2)))
    plt.plot(x, t, 'ko')        # 观测值的点集
    plt.plot(x2, probs, 'g-', label='raw fit')   # 拟合的sigmoid曲线
    plt.xlabel('Hours Studying')
    plt.ylabel('Probability of Passing Exam')
    plt.axis([-0.4, 6.2, -0.2, 1.1])
    plt.grid(True)
    plt.title('Logistic Regression Sample')

    plt.legend()
    plt.show()

# 取自维基百科的数据： https://en.wikipedia.org/wiki/Logistic_regression
Hours = np.r_[np.arange(0.5, 2, 0.25), np.arange(1.75, 3.75, 0.25), np.arange(4, 5.25, 0.25), 5.5]
Pass = np.r_[np.zeros(6), 1, 0, 1, 0, 1, 0, 1, 0, np.ones(6)]
x = Hours.reshape(-1, 1)
t = Pass.reshape(-1, 1)
print('training dataset: \n', x)
print('target variable: \n', t)
w = irls(x, t)
print('w calculated by myself: ', w)

# use sklearn for calculation
print('{0:-^70}'.format('Use sklearn to implement logistic regression'))
lr = slm.LogisticRegression(C=1.0)    # 这个模型强制性使用L1和L2的正则化，所以算出来的结果跟上面完全不一样
lr.fit(x, t)
print('w.intercept calculated by sklearn: ', lr.intercept_)
print('w.coefficients calculated by sklearn: ', lr.coef_)
print('Iter Counts: ', lr.n_iter_)

plt.subplot(111)
x3 = np.linspace(-0.3, 6, 500).reshape(-1, 1)
predicts = lr.predict_proba(x3)       # 预测了每种分类的概率，不只是1的概率，注意维度
log_predicts = lr.predict_log_proba(x3)
print(predicts.shape)
print(log_predicts)
plt.plot(x3, predicts[:, 1], 'y-', label='fit with regularization')
# plt.plot(x3, log_predicts[:, 1]/log_predicts[:, 0], 'b-', label='log_proba with regularization') # 类似于抛物线
plot(w, x, t)

    
