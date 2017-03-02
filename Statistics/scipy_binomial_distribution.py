import math
import numpy as np
import scipy.stats as sp

print('''
  Assume X ~ B(n, p);
  then E(X) = np, D(X) = npq = np(1 - p)
''')

n, p = 5, 0.4
mean, var, skew, kurt = sp.binom.stats(n, p, moments='mvsk')
print('Binomial Distribution: B({0}, {1})'.format(n, p))
print('Mean: ', mean)
print('Variance: ', var)
print('Skew: ', skew)
print('Kurt: ', kurt)

rv = sp.binom(n, p)
print('P(X = 0): ', rv.pmf(0))    # in continuous distribution, ppf is the same function
print('P(X = 1): ', rv.pmf(1))
print('P(X = 2): ', rv.pmf(2))
print('P(X = 3): ', rv.pmf(3))
print('P(X = 4): ', rv.pmf(4))
print('P(X = 5): ', rv.pmf(5)) 

print('{0:-^60}'.format('Seperate Line'))

print('P(X <= 1): ', rv.cdf(1))   # rv.pmf(0) + rv.pmf(1)
print('P(X <= 2): ', rv.cdf(2))
print('P(X <= 3): ', rv.cdf(3))
print('P(X <= 4): ', rv.cdf(4))
print('P(X <= 5): ', rv.cdf(5))

print('{0:-^60}'.format('Seperate Line'))

print('arg(P = 0.98976): X <= ', rv.ppf(0.98976))   # reverse function of cdf()
print('arg(P = 0.91296): X <= ', rv.ppf(0.91296))
print('arg(P = 0.68256): X <= ', rv.ppf(0.68256))
print('arg(P = 0.3): X <= ', rv.ppf(0.3))   # don't need to be accurate

print('''
  ----------------------------------
  entropy() function is used for calculating the entropy of a distribution for
  given probability values, for example, in ID3 decision tree algorithm.
  
  If only probabilities pk are given, the entropy is calculated as S = -sum(pk * log(pk), axis=0).

  If qk is not None, then compute the Kullback-Leibler divergence S = sum(pk * log(pk / qk), axis=0).
      base : float, optional
             The logarithmic base to use, defaults to e (natural logarithm).
''')

pk = [0.2, 0.8]
entropy = sp.entropy(pk, base=2)
validation = -(0.2*math.log(0.2, 2) + 0.8 * math.log(0.8, 2))
print('pk = ', pk)
print('entropy of pk: ', entropy)
print('validation: ', validation)
print(np.allclose(entropy, validation))
