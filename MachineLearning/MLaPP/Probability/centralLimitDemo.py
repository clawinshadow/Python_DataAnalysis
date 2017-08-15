import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

rv = ss.beta(1, 5)
samples = rv.rvs(10000)

plt.figure(figsize=(8, 5))
plt.subplot(121)
plt.title('N = 1')
plt.hist(samples, bins=25, color='darkblue', edgecolor='k')
plt.xticks([0, 0.5, 1])

samples_sum = samples
for i in range(4):
    samples_sum += rv.rvs(10000)

plt.subplot(122)
plt.title('N = 5')
plt.hist(samples_sum, bins=25, color='darkblue', edgecolor='k')
plt.xticks([0, 0.5, 1])

plt.show()
