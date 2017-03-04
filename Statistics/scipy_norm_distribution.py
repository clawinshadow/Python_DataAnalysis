import math
import numpy as np
import scipy.stats as sp

print('''
  Assume X ~ N(0, sigma),
  norm.pdf(x) = exp(-x**2/2)/sqrt(2*pi)
''')

mu, sigma = 0, 1
mean, var, skew, kurt = sp.norm.stats(mu, sigma, moments='mvsk')
print('Normal Distribution: B({0}, {1})'.format(mu, sigma))
print('Mean: ', mean)
print('Variance: ', var)
print('Skew: ', skew)
print('Kurt: ', kurt)

rv = sp.norm(mu, sigma)
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

print('arg(P = 0.975): X <= ', rv.ppf(0.975)) 
print('arg(P = 0.95): X <= ', rv.ppf(0.95))
print('arg(P = 0.90): X <= ', rv.ppf(0.90))
print('arg(P = 0.80): X <= ', rv.ppf(0.80)) 
print('arg(P = 0.70): X <= ', rv.ppf(0.70))
print('arg(P = 0.60): X <= ', rv.ppf(0.60))

print('{0:-^60}'.format('Seperate Line'))

print('sf(1.8): ', rv.sf(1.96))  # survival function(used in hypothesis)  

