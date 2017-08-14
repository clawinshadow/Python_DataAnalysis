import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

np.random.seed(0)

a = np.random.randn(30)
outliers = np.array([8, 8.75, 9.5])

plt.figure(figsize=(11, 5))
plt.subplot(121)

loc, scale = ss.norm.fit(a)
gauss = ss.norm(loc, scale)

loc, scale = ss.laplace.fit(a)
laplace = ss.laplace(loc, scale)

df, loc, scale = ss.t.fit(a)

plt.ylim(0, 0.5)
plt.xlim(-5, 10)
plt.hist(a, 7, weights=[1 / 30] * 30, rwidth=0.8)

x = np.linspace(-5, 10, 500)
plt.plot(x, gauss.pdf(x), ls=':', color='k', label='gaussian')
plt.plot(x, laplace.pdf(x), ls='--', color='b', label='laplace')
plt.plot(x, ss.t.pdf(x, df, loc, scale), color='r', label='student')
plt.legend()

plt.subplot(122)   # fit with outliers
plt.hist(a, 7, weights=[1 / 33] * 30, rwidth=0.8)
plt.hist(outliers, 3, weights=[1 / 33] * 3, color='b', rwidth=0.8)
aa = np.r_[a, outliers]

loc, scale = ss.norm.fit(aa)
gauss = ss.norm(loc, scale)

loc, scale = ss.laplace.fit(aa)
laplace = ss.laplace(loc, scale)

df, loc, scale = ss.t.fit(aa)
plt.ylim(0, 0.5)
plt.xlim(-5, 10)
plt.plot(x, gauss.pdf(x), ls=':', color='k', label='gaussian')
plt.plot(x, laplace.pdf(x), ls='--', color='b', label='laplace')
plt.plot(x, ss.t.pdf(x, df, loc, scale), color='r', label='student')
plt.legend()

plt.show()
