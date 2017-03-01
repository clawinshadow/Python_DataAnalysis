import numpy as np
import scipy.stats as sp

print('''
  from function side,
  np.max = np.amax = scipy.stats.tmax,
  np.min = np.amin = scipy.stats.tmin,
  ...
  ...
  ...
  but I found the stats module in scipy, it's the most powerful class to calculate
  statistic datas, so I use scipy in my demo mostly.

  -----------------------------------------------------------------

  ''')

def describe(data):
    a = np.asarray(data)
    if (a.size == 0):
        return None

    result = dict()
    result.setdefault('Data', a)
    result.setdefault('Mode', sp.stats.mode(a).mode)
    result.setdefault('Min', sp.stats.tmin(a))
    result.setdefault('Max', sp.stats.tmax(a))
    result.setdefault('Range', sp.stats.tmax(a) - sp.stats.tmin(a))
    result.setdefault('Variance', sp.stats.tvar(a))
    result.setdefault('Standard Variance', sp.stats.tstd(a))
    result.setdefault('Median', sp.stats.scoreatpercentile(a, 50))
    result.setdefault('High Quantile', sp.stats.scoreatpercentile(a, 75))
    result.setdefault('Low Quantile', sp.stats.scoreatpercentile(a, 25))
    result.setdefault('Arithmetic Mean', sp.stats.tmean(a))
    result.setdefault('Geometric Mean', sp.stats.gmean(a))
    result.setdefault('Harmonic Mean', sp.stats.hmean(a))
    result.setdefault('Skew', sp.stats.skew(a))
    result.setdefault('Kurtosis', sp.stats.kurtosis(a))
    result.setdefault('Coefficient of variation', sp.stats.tstd(a)/sp.stats.tmean(a))
    result.setdefault('Quantile Deviation', sp.stats.scoreatpercentile(a, 75) - sp.stats.scoreatpercentile(a, 25))
    
    return result;
    

a = np.arange(5)
print('np.max({0}): {1}'.format(a, np.max(a)))
print('np.amax({0}): {1}'.format(a, np.amax(a)))
print('sp.stats.tmax({0}): {1}'.format(a, sp.stats.tmax(a)))

print('{0:-^60}'.format('Seperate Line'))

data = [2, 4, 7, 10, 10, 10, 12, 12, 14, 15]
result = describe(data)
for k in result:
    print('{0:25}: {1}'.format(k, result[k]))


