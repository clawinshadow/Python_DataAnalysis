import math
import numpy as np
import scipy.stats as sp

print('''
  Assume X ~ Geo(p), it means in n-times Bernoulli trials, the first (n-1) times
  all failed until the last time, so:
  P(X = n) = (1-p)**(n-1)*p
  E(X) = 1/p, D(X) = (1-p)/p**2
''')


p = 0.2
mean, var, skew, kurt = sp.geom.stats(p, moments='mvsk')
print('Geometric Distribution: G({0})'.format(p))
print('Mean: ', mean)
print('Variance: ', var)
print('Skew: ', skew)
print('Kurt: ', kurt)

print('{0:-^60}'.format('Seperate Line'))

rv = sp.geom(p)
print('P(X = 1): ', rv.pmf(1))  # 0.2
print('P(X = 2): ', rv.pmf(2))  # 0.8 * 0.2
print('P(X = 3): ', rv.pmf(3))  # 0.8 ** 2 * 0.2
print('P(X = 4): ', rv.pmf(4))
print('P(X = 5): ', rv.pmf(5)) 

print('{0:-^60}'.format('Seperate Line'))

print('P(X <= 1): ', rv.cdf(1))   # rv.pmf(0) + rv.pmf(1)
print('P(X <= 2): ', rv.cdf(2))
print('P(X <= 3): ', rv.cdf(3))
print('P(X <= 4): ', rv.cdf(4))
print('P(X <= 5): ', rv.cdf(5))

print('{0:-^60}'.format('Seperate Line'))

print('arg(P = 0.95): X <= ', rv.ppf(0.98976))   # reverse function of cdf()
print('arg(P = 0.80): X <= ', rv.ppf(0.91296))
print('arg(P = 0.70): X <= ', rv.ppf(0.68256))
print('arg(P = 0.60): X <= ', rv.ppf(0.3))
