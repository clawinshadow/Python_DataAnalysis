import math
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt

print('''
  设随机变量X1, X2, X3, ..., Xn相互独立，且Xi均服从标准正态分布N(0, 1),
  则他们的平方和Σ(Xi)**2服从自由度为 n 的χ2分布
  
  Assume X ~ χ2(df),
  chi2.pdf(x, df) = 1 / (2*gamma(df/2)) * (x/2)**(df/2-1) * exp(-x/2)
  E(X) = n, D(X) = 2n

  并且χ2分布具有可加性，若(X1)**2 ~ χ2(n1), (X2)**2 ~ χ2(n2),
  则 (X1)**2 + (X2)**2 ~ χ2(n1 + n2)
''')

df = 3  # degree of freedom (自由度)
mean, var, skew, kurt = sp.chi2.stats(df, moments='mvsk')
print('Chi-square Distribution: χ2({0})'.format(df))
print('Mean: ', mean)
print('Variance: ', var)
print('Skew: ', skew)
print('Kurt: ', kurt)

rv = sp.chi2(df)
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
print('P(X <= 0.1): ', rv.pdf(0.1)) # pdf means probability density function
print('P(X <= 0.4): ', rv.pdf(0.4))
print('P(X <= 0.8): ', rv.pdf(0.8))  
print('P(X <= 1): ', rv.pdf(1))
print('P(X <= 2): ', rv.pdf(2))
print('P(X <= 3): ', rv.pdf(3))

print('{0:-^60}'.format('Seperate Line'))
# 卡方分布一般是个右偏分布，自由度越大越近似于正态分布， n = 1 时特殊情况
print('arg(P = 0.975): X <= ', rv.ppf(0.975))   
print('arg(P = 0.95): X <= ', rv.ppf(0.95))
print('arg(P = 0.90): X <= ', rv.ppf(0.90))
print('arg(P = 0.80): X <= ', rv.ppf(0.80)) 
print('arg(P = 0.70): X <= ', rv.ppf(0.70))
print('arg(P = 0.60): X <= ', rv.ppf(0.60))

print('{0:-^60}'.format('Seperate Line'))

print('sf(9.3484): ', rv.sf(9.3484))  # survival function(used in hypothesis)

df = 20
rv = sp.chi2(df)
fig, ax = plt.subplots(1, 1)
x = np.linspace(rv.ppf(0.01), rv.ppf(0.99), 100)
ax.plot(x, rv.pdf(x), 'r-', lw=5, alpha=0.6, label='chi2 pdf')
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
r = rv.rvs(1000)
ax.hist(r, normed=True, histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)
plt.show()

