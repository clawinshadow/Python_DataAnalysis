import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

x = np.arange(0, 11, 1)
rv1 = ss.binom(10, 0.25) # 二项分布 Bin(10, 0.25)
rv2 = ss.binom(10, 0.9)  # 二项分布 Bin(10, 0.9)
y1 = rv1.pmf(x)
y2 = rv2.pmf(x)

plt.figure(figsize=(11, 5))
plt.subplot(121)
plt.bar(x, y1, align='center')
plt.ylim(0, 0.35)
plt.xticks(x)
plt.xlim(-1, 11)
plt.title(r'$\theta=0.25$')

plt.subplot(122)
plt.bar(x, y2, align='center')
plt.ylim(0, 0.35)
plt.xticks(x)
plt.xlim(-1, 11)
plt.title(r'$\theta=0.90$')

plt.show()
