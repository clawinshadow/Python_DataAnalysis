import math
import numpy as np

'''
Cov means Covariance, 即协方差
Covariance indicates the level to which two variables vary together
用来衡量两个变量之间一起变化的程度
假设随机向量X有p个分量，每个分量有 n 个观测值，则X可表示为一个 n*p 的矩阵
那么Cov(X)应该是 p*p 的方阵，两两比较不同维度之间的n个观测值
但是numpy.cov默认的是rowvar=True，即每一行为一个维度，要注意这一点

回顾几个概念：
Variance(方差):
- Definition: Var(X) = E(X**2) - E(X)**2 = E{[X - E(X)]**2}
- Properties: Var(c) = 0, Var(cX) = c**2 * Var(X), Var(X + c) = Var(X)

Covariance:
- Definition: Cov(X, Y) = E(XY) - E(X)E(Y) = E[X-E(X)][Y-E(Y)]
- Properties:
        * Symmetry: Cov(X, Y) = Cov(Y, X)
        * Relation to variance: Cov(X, X) = Var(X), Var(X + Y) = Var(X) + Var(Y) + 2Cov(X, Y)
        * Bilinearity: Cov(cX, Y) = Cov(X, cY) = c * Cov(X, Y)
                       Cov(X1 + X2, Y) = Cov(X1, Y) + Cov(X2, Y)
                       Cov(X, Y1 + Y2) = Cov(X, Y1) + Cov(X, Y2)

Correlation:
                                 Cov(X, Y)
- Definition: ρ(X, Y) = ----------------------------
                         math.sqrt(Var(X) * Var(Y))
- Properties: -1 <= ρ(X, Y) <= 1
        * 相关阵的对角线元素均为1
        * 在数据处理时，为了克服由于指标的量纲不同对统计分析造成的影响，往往事先将每个数据标准化：
                    x - E(X)
          x = --------------------
                math.sqrt(Var(X))
          于是对每一维的分量Xi, i = 1, 2, 3, ..., p
          E(X) = 0
          D(X) = corr(X) = R
          即对于标准化后的数据来说，协方差阵正好是相关阵
'''

x = np.array([-2.1, -1, 4.3])
y = np.array([3, 1.1, 0.12])
X = np.vstack((x, y))
print('X: \n', X)

Ex = np.average(x)
Ex2 = np.average(x**2)
Varx = Ex2 - Ex**2
Ey = np.average(y)
Ey2 = np.average(y**2)
Vary = Ey2 - Ey**2
print('Varx: {0}, Vary: {1}'.format(Varx, Vary))

Exy = np.average(x*y)
Covxy = Exy - Ex * Ey
CovX = np.array([[Varx, Covxy], [Covxy, Vary]])
print('CovX: \n', CovX)
# bias参数用来控制标准化方差的时候，是除以 n - 1 还是 n, 默认是False，即除以 n - 1的
# 本例中观测值数量为3， 所以np.cov默认是用2来标准化方差的，设为True后才与手算的match
print('CovX == np.cov(X): ', np.allclose(CovX, np.cov(X, bias=True)))

# 相关阵
Corrxy = Covxy / math.sqrt(Varx * Vary)
R = np.array([[1, Corrxy], [Corrxy, 1]])
print('R: \n', R)
print('R == np.corrcoef(X, bias=True):', np.allclose(R, np.corrcoef(X, bias=True)))



