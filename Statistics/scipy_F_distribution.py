import math
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt

print('''
  设随机变量 Y 与 Z相互独立，且Y和Z分别服从自由度为m和n的χ2分布，
  随机变量X有如下表达式：
  X = (Y/m)/(Z/n) = nY/mZ
  则称X服从第一自由度为m，第二自由度为n的F分布，记为F(m, n)
  
  Assume X ~ T(df),
                        df2**(df2/2) * df1**(df1/2) * x**(df1/2-1)
  F.pdf(x, df1, df2) = --------------------------------------------
                       (df2+df1*x)**((df1+df2)/2) * B(df1/2, df2/2)

  E(t) = n/(n-2), when n > 2
  D(t) = 2*n**2*(m+n-2)/m(n-2)(n-4), when n > 4.
  
  两个自由度的位置不可互换，另外:
  如果随机变量X服从t(n)分布，则X**2服从F(1, n)
''')

dfm, dfn = 5, 10  # degree of freedom (自由度)
mean, var, skew, kurt = sp.f.stats(dfm, dfn, moments='mvsk')
print('F Distribution: F({0}, {1})'.format(dfm, dfn))
print('Mean: ', mean)
print('Variance: ', var)
print('Skew: ', skew)
print('Kurt: ', kurt)

rv = sp.f(dfm, dfn)
print('P(X <= 0.1): ', rv.cdf(0.1))
print('P(X <= 0.4): ', rv.cdf(0.4))
print('P(X <= 0.8): ', rv.cdf(0.8))  
print('P(X <= 1): ', rv.cdf(1))
print('P(X <= 2): ', rv.cdf(2))
print('P(X <= 3): ', rv.cdf(3))
print('P(X <= 4): ', rv.cdf(4))
print('P(X <= 5): ', rv.cdf(5))
print('P(X <= 6): ', rv.cdf(6))

print('{0:-^60}'.format('Seperate Line'))

print('probability density: ')
print('P(X <= 0.1): ', rv.pdf(0.1))
print('P(X <= 0.4): ', rv.pdf(0.4))
print('P(X <= 0.8): ', rv.pdf(0.8))  
print('P(X <= 1): ', rv.pdf(1))
print('P(X <= 2): ', rv.pdf(2))
print('P(X <= 3): ', rv.pdf(3))

print('{0:-^60}'.format('Seperate Line'))
print('arg(P = 0.975): X <= ', rv.ppf(0.975))   
print('arg(P = 0.95): X <= ', rv.ppf(0.95))
print('arg(P = 0.90): X <= ', rv.ppf(0.90))
print('arg(P = 0.80): X <= ', rv.ppf(0.80)) 
print('arg(P = 0.70): X <= ', rv.ppf(0.70))
print('arg(P = 0.60): X <= ', rv.ppf(0.60))

print('{0:-^60}'.format('Seperate Line'))

print('sf(4.236): ', rv.sf(4.236))  # survival function(used in hypothesis)  

#图形上类似于卡方分布
fig, ax = plt.subplots(1, 1)
x = np.linspace(rv.ppf(0.01), rv.ppf(0.99), 100)
ax.plot(x, rv.pdf(x), 'r-', lw=5, alpha=0.6, label='F pdf')
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
r = rv.rvs(1000)
ax.hist(r, normed=True, histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)
plt.show()

