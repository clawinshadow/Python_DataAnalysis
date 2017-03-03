import math
import numpy as np
import scipy.stats as sp


print('''
The hypergeometric distribution models drawing objects from a bin.
M is the total number of objects, n is total number of Type I objects.
The random variate represents the number of Type I objects in N
drawn WITHOUT replacement from the total population,
while in binomial distribution, it's WITH replacement

pmf(k, M, n, N) = choose(n, k) * choose(M - n, N - k) / choose(M, N),
                  for max(0, N - (M-n)) <= k <= min(n, N)
''')


[M, n, N] = [20, 7, 12]
mean, var, skew, kurt = sp.hypergeom.stats(M, n, N, moments='mvsk')
print('Hyper-geometric Distribution: HyperGeo({0}, {1}, {2})'.format(M, n, N))
print('Mean: ', mean)
print('Variance: ', var)
print('Skew: ', skew)
print('Kurt: ', kurt)

print('{0:-^60}'.format('Seperate Line'))

rv = sp.hypergeom(M, n, N)
print('P(X = 0): ', rv.pmf(0))    # C(12, 13) / C(12, 20)
print('P(X = 1): ', rv.pmf(1))    # C(1, 7) * C(11, 13) / C(12, 20)
print('P(X = 2): ', rv.pmf(2))
print('P(X = 3): ', rv.pmf(3))
print('P(X = 4): ', rv.pmf(4))
print('P(X = 5): ', rv.pmf(5))
print('P(X = 6): ', rv.pmf(6)) 
print('P(X = 7): ', rv.pmf(7)) 

print('{0:-^60}'.format('Seperate Line'))

print('P(X <= 1): ', rv.cdf(1))   # rv.pmf(0) + rv.pmf(1)
print('P(X <= 2): ', rv.cdf(2))
print('P(X <= 3): ', rv.cdf(3))
print('P(X <= 4): ', rv.cdf(4))
print('P(X <= 5): ', rv.cdf(5))
print('P(X <= 6): ', rv.cdf(6))
print('P(X <= 7): ', rv.cdf(7))

print('{0:-^60}'.format('Seperate Line'))

print('arg(P = 0.9): X <= ', rv.ppf(0.98976))   # reverse function of cdf()
print('arg(P = 0.7): X <= ', rv.ppf(0.91296))
print('arg(P = 0.5): X <= ', rv.ppf(0.68256))
print('arg(P = 0.3): X <= ', rv.ppf(0.3))   # don't need to be accurate
