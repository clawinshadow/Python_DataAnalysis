import numpy as np
import scipy.stats as st
import math
from sklearn import linear_model
import matplotlib.pyplot as plt

'''
主要包括一元线性回归的相关统计分析
估计的回归方程：y = beta0 + beta1 * x
两个beta系数的理论基础为最小二乘法，可由微分极值定理或矩阵理论推出

回归直线的拟合优度：yi表示观测值，Ey表示回归估计值，mean(y)表示观测均值
    SST: n次观测值的总变差, sigma(yi - mean(y))**2
    SSR: 回归值减去观测均值的平方和， sigma(Ey - mean(y))**2
    SSE: 回归值减去观测值的平方和， sigma(Ey - yi)**2

    SSR反映了y的总变差中由x与y之间的线性关系引起的y的变化部分，是可由
    回归直线解释的部分，记为回归平方和；
    SSE表示不能由回归直线解释的部分，称为残差平方和或误差平方和

    判定系数R Square = SSR / SST
    该值越大，表明回归直线拟合的越好
    估计标准误差：Se = math.sqrt(SSE/(n-2))

显著性检验：
1. 线性关系的检验：检验两个变量之间的线性关系是否显著

    F = (SSR/1)/(SSE/(n-2)) = MSR/MSE ~ F(1, n-2)
    若 F > F crit 则线性关系显著

2. 回归系数的检验：检验回归系数beta1是否等于0，等于0的话表明回归线是一条
                   水平线，y和x之间没有线性关系

    Sbeta = Se / math.sqrt(sigma(xi**2) - sigma(xi)**2 / n)

    t = beta1 / Sbeta ~ t(n-2)

在一元线性回归中，F检验和T检验是等价的，但是在多元线性回归中
这两个的意义的是不同的，F检验只是用来检验总体回归关系的显著性
但是t检验用来检验每个回归系数的显著性
'''

def line(x, beta0, beta1):
    return beta0 + beta1 * x

def pearson(x, y):
    '''
    求两个数值型变量之间的相关系数，度量线性关系强度
    '''
    if len(x) != len(y):
        raise ValueError('The size of x and y must be the same')

    n = len(x)
    x = np.array(x)
    y = np.array(y)
    sumxy = np.sum(x * y)
    sumx = np.sum(x)
    sumy = np.sum(y)
    sumx2 = np.sum(x**2)
    sumy2 = np.sum(y**2)

    divisor = math.sqrt(n * sumx2 - sumx **2) * math.sqrt(n * sumy2 - sumy **2)
    pearson = (n * sumxy - sumx * sumy ) / divisor
    print('pearson: ', pearson)
    return pearson


def linear_regression(x, y, alpha, x0=0):
    '''
    用最小二乘法进行一元线性回归
    
    Parameters
    ----------
    x, y : 两个长度一致的数值型数组
    alpha: 执行假设检验的置信度
    x0: 对一个给定值进行y的平均值&个别值的区间估计
    
    Returns
    -------
    R: 自变量与因变量之间的关系强度， R**2 = SSR + SSC / SST
    
    '''
    if len(x) != len(y):
        raise ValueError('The size of x and y must be the same')

    # Phase 1: 计算回归方程
    n = len(x)
    x = np.array(x)
    y = np.array(y)
    sumxy = np.sum(x * y)
    sumx = np.sum(x)
    sumy = np.sum(y)
    sumx2 = np.sum(x**2)
    meanx = np.mean(x)
    meany = np.mean(y)

    beta1 = ((n * sumxy - sumx * sumy) / (n * sumx2 - sumx**2))
    beta0 = (meany - meanx * beta1)
    print('estimated regression equation: y = {0} + {1} * x'.format(beta0.round(6), beta1.round(6)))

    # Phase 2：计算拟合优度和估计标准差
    vfunc = np.vectorize(line)  # 向量化回归方程的计算
    sst = np.var(y) * n
    Ey = vfunc(x, beta0, beta1)
    ssr = np.sum((Ey - meany) ** 2)
    sse = np.sum((Ey - y) ** 2)
    # print(sst, ssr, sse)

    RSquare = ssr/sst
    Se = math.sqrt(sse / (n - 2))
    print('R Square: {0}, 标准误差: {1}'.format(RSquare, Se))

    # Phase 3：对回归方程进行显著性检验
    mse = sse / (n - 2)
    msr = ssr / 1
    F = msr / mse
    rv = st.f(1, n-2)
    Fcrit = rv.ppf(1 - alpha)
    if (F > Fcrit):
        print('F({0}) > FCrit({1}), 线性关系显著'.format(F, Fcrit))
    else:
        print('F({0}) <= FCrit({1}), 线性关系不显著'.format(F, Fcrit))

    sbeta = Se / math.sqrt(sumx2 - sumx**2/n)
    t = beta1 / sbeta
    rv = st.t(n-2)
    tcrit = rv.ppf(1 - alpha/2)
    if (t > tcrit):
        print('t({0}) > tcrit({1}), 回归系数显著'.format(t, tcrit))
    else:
        print('t({0}) <= tcrit({1}), 回归系数不显著'.format(t, tcrit))

    # Phase 4: 对一个给定值x0，分别预测y的平均值与个别值的区间估计
    Smean = Se * math.sqrt(1/n + (x0-meanx)**2 / np.sum((x-meanx)**2))
    tmean = tcrit * Smean
    y0 = line(x0, beta0, beta1)
    ymean_estimate = y0 - tmean, y0 + tmean

    Sind = Se * math.sqrt(1 + 1/n + (x0-meanx)**2 / np.sum((x-meanx)**2))
    tind = tcrit * Sind
    yind_estimate = y0 - tind, y0 + tind

    print('To x0 = ', x0)
    print('y的平均值的置信区间估计：', ymean_estimate)
    print('y的个别值的置信区间估计：', yind_estimate)

    print('{0:-^90}'.format(''))
    print('{0:^15}|{1:^15}|{2:^10}|{3:^15}|{4:^15}|{5:^15}'.format(\
        'Source', 'SS', 'df', 'MS', 'F', 'F crit'))
    print('{0:-^85}'.format(''))
    print('{0:^15}|{1:^15}|{2:^10}|{3:^15}|{4:^15}|{5:^15}'.format(\
        'Regression', ssr.round(6), 1, msr.round(6), F.round(6), Fcrit.round(6))) # 回归
    print('{0:^15}|{1:^15}|{2:^10}|{3:^15}|'.format(\
        'Residual', sse.round(6), n-2, mse.round(6))) # 残差
    print('{0:^15}|{1:^15}|{2:^10}|'.format('Total', sst.round(6), n-1)) # 总计


def linear_regression_brief(x, y):
    '''
    使用sklearn.linear_model.LinearRegression来实现最小二乘法的线性回归

    Parameter: x 必须是一个R(m*n)矩阵，并且m > n, 必须是超定的矩阵
               y 是一个一维数组，长度与m一致，为观测值集合
    
    Docs:http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    '''
    reg = linear_model.LinearRegression()
    x = np.array(x)
    x = x[:, np.newaxis]
    reg.fit(x, y)
    print('coefficients: ', reg.coef_)
    print('intercept: ', reg.intercept_)
    # print('residues: ', reg.residues_)    # deprecated
    print('R Square: ', reg.score(x, y))

    plt.scatter(x, y, color='blue')
    plt.plot(x, reg.predict(x), color='green', linewidth=3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    

y = [0.9,1.1,4.8,3.2,7.8,2.7,1.6,12.5,1.0,2.6,0.3,4.0,0.8,3.5,10.2,3.0,\
     0.2,0.4,1.0,6.8,11.6,1.6,1.2,7.2,3.2]
x = [67.3,111.3,173,80.8,199.7,16.2,107.4,185.4,96.1,72.8,64.2,132.2,58.6,\
     174.6,263.5,79.3,14.8,73.5,24.7,139.4,368.2,95.7,109.6,196.2,102.2]
pearson(x, y)
linear_regression(x, y, 0.05, x0=100)
linear_regression_brief(x, y)
