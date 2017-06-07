import numpy as np
import sklearn.linear_model as slm

'''
感知机，一个历史地位很高的算法，算是模式识别领域的一个里程碑算法，经历过辉煌和被冷落的阶段，虽然连异或问题都
搞不定，但依然有划时代的意义。与大部分广义线性模型一样，依然是给定一个输入向量x，再给定一个固定的基函数(通常
是非线性的)，来将输入向量x转化为Φ(x), 最后代入广义线性模型：

    y(x, w) = f(w.T*Φ(x))

这里激活函数f(*)是一个离散的阶跃函数，PRML里面习惯用[+1, -1], 而不是用[0, 1]作为分类变量

            +1,   a >= 0
    f(a) =
            -1,   a  < 0

定义目标变量tn值为+1的属于分类C1，否则属于分类C2，那么按照一般的方法，我们要定义个损失函数，然后通过求解它的
最小值来计算出权重向量 w，如果使用最直接的0/1损失函数，那么对于每个可能的w来说，损失都是一个整数，显然是个离
散的函数，也不可能使用随机梯度下降算法来寻找最优解了，所以我们得设计出一个连续的数学性质友好的损失函数如下，
这个规则也称之为感知机准则，perceptron criterion

假设一个样本被错误的预测为C1了，表明w.T*Φ(x)是大于零的，而它对应的tn却是-1，这时损失w.T*Φ(x)*tn < 0，相应的，
如果一个样本被错误的预测为C2，则表明w.T*Φ(x) < 0，而tn = +1，所以这时损失w.T*Φ(x)*tn也小于零，w.T*Φ(x)*tn的
绝对值越小，表明总的损失越小，如果所有分类都是对的，那么损失应该为零，基于这些条件，

我们给定如下的损失函数： E(w) = -Σ[w.T*Φ(x)*tn], n的范围仅限于错误分类的样本

这个函数就是连续的，piecewise linear, 片段线性的？几何意义上就是由一个个倾斜的斜面衔接起来的曲面，使用随机梯
度下降法来求解E(w)的最小值

    w.new = w.old - η*D[E(w)].T = w.old + η*Φ(x)*tn

因为 w 乘以一个常数不会影响f(a)的结果，所以不失一般性，我们定义学习率为1，那么对于上面的公式，因为tn只能取值
+1或者-1，所以可以理解为当一个样本被错误的分类为C1时，tn=1，则是用当前的权重加上当前的样本向量作为新的权重向
量，反之，则是减去当前的样本向量

由感知机收敛理论可知，只要训练数据集是线性可分的，那么感知机算法一定收敛，并且根据初始参数的不同，或者样本数
据点的顺序不一样，最终得出来的w也会是不一样的，它没有唯一解

最后，就像所有逐行样本遍历的迭代算法一样，它既支持批量的离线学习也支持一个接一个过来的在线即时学习
'''

def reweight(w, xn, tn):
    if tn == '+1':
        return w + xn
    else:
        return w - xn

def activate_func(w, x):
    flag = np.dot(w.T, x)  
    if flag >= 0:
        return '+1'
    else:
        return '-1'

# 遍历一遍训练数据集，如果权重向量w没有任何改变，则说明已经收敛成功
def one_cycle(w, x, t, count):
    rows, cols = x.shape
    for i in range(rows):
        xi = x[i].reshape(-1, 1)
        ti = t[i]               # scalar, +1 or -1
        flag = activate_func(w, xi)
        count += 1
        if flag == ti:
            print('Iter Count {0}: Classified Correctly'.format(count))
        else:
            w = reweight(w, xi, ti)
            print('Iter Count {0}: update w to {1}'.format(count, w.ravel()))
            
    return w, count
            
def iterate(x, t, n_iter=1):
    x = np.c_[np.ones(x.shape[0]).reshape(-1, 1), x]
    rows, cols = x.shape
    w_old = np.zeros(cols).reshape(-1, 1)
    count = 0
    while True:
        w_new, count = one_cycle(w_old, x, t, count)
        if np.allclose(w_old, w_new):
            return w_new
        else:
            w_old = w_new

def predict(w, x):
    x = np.c_[np.ones(x.shape[0]).reshape(-1, 1), x]
    flag = np.dot(x, w)
    return np.where(flag >= 0, '+1', '-1')

x = np.array([[0.1, 1.15],
              [0.23, 1.55],
              [0.35, 1.25],
              [0.52, 1.75],
              [0.67, 1.3],
              [0.8, 1.9],
              [0.95, 1.8],
              [1.05, 0.15],
              [1.17, 0.95],
              [1.31, 0.2],
              [1.42, 0.75],
              [1.49, 0.52],
              [1.75, 0.25],
              [1.91, 0.83]])
t = np.array(['+1', '+1', '+1', '+1', '+1', '+1', '+1', '-1', '-1', '-1', '-1', '-1', '-1', '-1'])
w = iterate(x, t)
print('x: \n', x)
print('t: ', t)
print('w by perceptron: ', w.ravel())
print('predict x: ', predict(w, x).ravel())

# use sklearn for calculation
print('{0:-^70}'.format('Use sklearn to implement perceptron'))
p = slm.Perceptron(n_iter=3, shuffle=False)  # 遍历三次i，每次遍历前不打乱数据，与我手算的程序一致
print('Perceptron class: ', p)
p.fit(x, t)
print('p.intercept_: ', p.intercept_)
print('p.coef_: ', p.coef_.ravel())
print('predict x: ', p.predict(x))
print('score always equal to 1.0: ', p.score(x, t))
