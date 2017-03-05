import math
import numpy as np
import scipy.stats as sp

print('''
  设随机变量 X ~ N(0, 1), Y ~ χ2(n), 且X与Y独立，则
  t = X /sqrt(Y/n) ~ t(n), 主要用于小样本理论
  
  Assume X ~ T(df),
                                 gamma((df+1)/2)
  t.pdf(x, df) = ---------------------------------------------------
                 sqrt(pi*df) * gamma(df/2) * (1+x**2/df)**((df+1)/2)

  E(t) = 0, when n > 1
  D(t) = n/(n-2), when n > 2
  
  To shift and/or scale the distribution use the loc and scale parameters.
  Specifically, t.pdf(x, df, loc, scale) is identically equivalent to
  t.pdf(y, df) / scale with y = (x - loc) / scale.
''')

df = 5  # degree of freedom (自由度)
mean, var, skew, kurt = sp.t.stats(df, moments='mvsk')
print('Student\'s Distribution: T({0})'.format(df))
print('Mean: ', mean)
print('Variance: ', var)
print('Skew: ', skew)
print('Kurt: ', kurt)

rv = sp.t(df)
print('P(X <= -1): ', rv.cdf(-1))
print('P(X <= -2): ', rv.cdf(-2))
print('P(X <= -3): ', rv.cdf(-3))
print('P(X <= 0): ', rv.cdf(0))   
print('P(X <= 1): ', rv.cdf(1))
print('P(X <= 2): ', rv.cdf(2))
print('P(X <= 3): ', rv.cdf(3))

print('{0:-^60}'.format('Seperate Line'))

print('probability density: ')
print('f(X = -1): ', rv.pdf(-1))
print('f(X = -2): ', rv.pdf(-2))
print('f(X = -3): ', rv.pdf(-3))
print('f(X = 0): ', rv.pdf(0))   
print('f(X = 1): ', rv.pdf(1))
print('f(X = 2): ', rv.pdf(2))
print('f(X = 3): ', rv.pdf(3))

print('{0:-^60}'.format('Seperate Line'))
# T分布比正态分布更扁平一点，所以同样的概率，T分布的值要比正态分布的更大一点
print('arg(P = 0.975): X <= ', rv.ppf(0.975))   
print('arg(P = 0.95): X <= ', rv.ppf(0.95))
print('arg(P = 0.90): X <= ', rv.ppf(0.90))
print('arg(P = 0.80): X <= ', rv.ppf(0.80)) 
print('arg(P = 0.70): X <= ', rv.ppf(0.70))
print('arg(P = 0.60): X <= ', rv.ppf(0.60))

print('{0:-^60}'.format('Seperate Line'))

print('sf(1.8): ', rv.sf(1.96))  # survival function(used in hypothesis)  

