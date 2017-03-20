import math
import numpy as np
import scipy.stats as st
from sklearn import linear_model

'''
多元线性回归，理论基础与一元的一样，但是这里的代码主要用矩阵理论来实现，
偏导数的不太好写，再就是对各系数做显著性检验，检查多元共线性等问题

估计的回归方程: y = beta0 + beta1 * x1 + beta2 * x2 + ... + betan * xn
SST, SSR, SSE, 判定系数R Square的概念和计算方式与一元线性回归中的一致
但是多了个调整的R Square，将自变量的个数也考虑进去了

Ra Square = 1 - (1 - R Square) * (n - 1) / (n - k - 1)

估计标准误差：Se = math.sqrt(SSE/(n-k-1))

线性关系检验：

    F = SSR/k/SSE/(n-k-1) ~ F(k, n-k-1)

回归系数检验：与一元线性回归不一样，多元的需要对每个回归系数都做检验

    Sbetai = Se / math.sqrt(sigma(xi**2) - sigma(xi)**2 / n)

    t = betai / Sbetai ~ t(n-k-1)

多重共线性的判别：
    1. 计算每两个自变量之间的相关系数矩阵
    2. 构造每个相关系数的t检验量

       t = |r|*math.sqrt((n-2)/(1-r**2)) ~ t(n-2)

'''

def linear_regression(x, y, alpha):
    '''
    估计多元线性回归的方程，并进行各种统计分析
    '''
    # Phase 1: 估计回归方程
    reg = linear_model.LinearRegression()
    x = np.array(x)
    print(x.shape)
    m, n = x.shape
    if m < n and n == len(y):
        x = np.transpose(x)
    
    reg.fit(x, y)
    print('coefficients: ', reg.coef_)
    print('intercept: ', reg.intercept_)

    # Phase 2：计算判定系数和标准误差等
    Ey = reg.predict(x)
    meany = np.mean(y)
    sst = np.sum((y - meany)**2)
    ssr = np.sum((Ey - meany)**2)
    sse = np.sum((Ey - y)**2)

    n, k = x.shape
    Rsquare = ssr/sst # 判定系数
    Rasquare = 1 - (1 - Rsquare) * (n - 1) / (n - k - 1) # 调整的判定系数
    Se = math.sqrt(sse / (n - k - 1)) # 估计的标准误差
    print('R Square: ', Rsquare)
    print('Adjusted R Square: ', Rasquare)
    print('Se: ', Se)
    
    # Phase 3：显著性检验
    msr = ssr / k
    mse = sse / (n - k - 1)
    F = msr / mse
    rv = st.f(k, n-k-1)
    Fcrit = rv.ppf(1 - alpha)  # 线性关系显著性检验
    if (F > Fcrit):
        print('F({0}) > FCrit({1}), 线性关系显著'.format(F, Fcrit))
    else:
        print('F({0}) <= FCrit({1}), 线性关系不显著'.format(F, Fcrit))

    rv = st.t(n-k-1)
    tcrit = rv.ppf(1 - alpha/2) 
    for i in range(k):
        xi = x[:, i]
        sumx = np.sum(xi)
        sumx2 = np.sum(xi**2)
        sbetai = Se / math.sqrt(sumx2 - sumx**2/n)
        t = reg.coef_[i] / sbetai
        
        if (abs(t) > abs(tcrit)):
            print('|t|({0}) > |tcrit|({1}), 回归系数beta{2}显著'.format(t, tcrit, i+1))
        else:
            print('|t|({0}) <= |tcrit|({1}), 回归系数beta{2}不显著'.format(t, tcrit, i+1))
    
    # Phase 4: 多重共线性的判别
    # 计算每两个自变量之间的相关性矩阵，及t检验量
    print('{0:-^100}'.format(''))
    print('{0:^15}|{1:^20}|{2:^20}|{3:^20}|{4:^20}'.format(\
        'tcrit:'+str(tcrit.round(4)), 'x1', 'x2', 'x3', 'x4'))
    print('{0:-^100}'.format(''))
    for i in range(k):
        pearsons = []
        for j in range(k):
            if i == j:
                pearsons.append(str(1.0))
            else:
                r = np.round(st.pearsonr(x[:, i], x[:, j])[0], 6)
                t = np.round(abs(r) * math.sqrt((n - 2) / (1 - r**2)), 6)
                pearsons.append('{0}({1})'.format(r, t))
        print('{0:^15}|{1:^20}|{2:^20}|{3:^20}|{4:^20}'.format(\
            'x'+str(i+1), pearsons[0], pearsons[1], pearsons[2], pearsons[3] ))
    
            
x1 = [67.3,111.3,173,80.8,199.7,16.2,107.4,185.4,96.1,72.8,64.2,132.2,58.6,\
     174.6,263.5,79.3,14.8,73.5,24.7,139.4,368.2,95.7,109.6,196.2,102.2]
x2 = [6.8,19.8,7.7,7.2,16.5,2.2,10.7,27.1,1.7,9.1,2.1,11.2,6,12.7,15.6,8.9,\
      0.6,5.9,5.0,7.2,16.8,3.8,10.3,15.8,12.0]
x3 = [5,16,17,10,19,1,17,18,10,14,11,23,14,26,34,15,2,11,4,28,32,10,14,16,10]
x4 = [51.9,90.9,73.7,14.5,63.2,2.2,20.2,43.8,55.9,64.3,42.7,76.7,22.8,\
      117.1,146.7,29.9,42.1,25.3,13.4,64.3,163.9,44.5,67.9,39.7,97.1]
y = [0.9,1.1,4.8,3.2,7.8,2.7,1.6,12.5,1.0,2.6,0.3,4.0,0.8,3.5,10.2,3.0,0.2,\
     0.4,1.0,6.8,11.6,1.6,1.2,7.2,3.2]

linear_regression([x1,x2,x3,x4], y, 0.05)
