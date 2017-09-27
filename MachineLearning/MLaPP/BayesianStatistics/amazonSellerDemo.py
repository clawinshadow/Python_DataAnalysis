import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

np.random.seed(0)

s1 = ss.beta(91, 11)
s2 = ss.beta(3, 1)

x = np.linspace(0.001, 0.999, 100)

fig = plt.figure(figsize=(11, 5))
fig.canvas.set_window_title('amazonSellerDemo')

plt.subplot(121)
plt.axis([0, 1, 0, 14])
plt.xticks(np.linspace(0, 1, 11))
plt.yticks(np.linspace(0, 14, 8))
plt.plot(x, s1.pdf(x), 'r-', lw=2, label=r'$p(\theta_1|data)$')
plt.plot(x, s2.pdf(x), 'g:', lw=2, label=r'$p(\theta_2|data)$')
plt.legend(loc='upper left')

N = 20000 # MC sample count
s1_sample = s1.rvs(N)
s2_sample = s2.rvs(N)
gap = s1_sample - s2_sample
ratio_MC = np.count_nonzero(gap > 0) / N
ratio = 0.710  # 书里面给出的真实值
q = np.percentile(gap, [2.5, 97.5])
print(q)
kde = ss.gaussian_kde(gap)
x2 = np.linspace(-0.4, 1, 200)
y2 = kde.pdf(x2)

plt.subplot(122)
plt.title(r'$prob(\theta_1 > \theta_2|D): MC={0:.3}, exact={1:.3}$'.format(ratio_MC, ratio))
plt.xlabel(r'$\delta$')
plt.ylabel('pdf')
plt.axis([-0.4, 1, 0, 2.5])
plt.xticks(np.linspace(-0.4, 1, 8))
plt.yticks(np.linspace(0, 2.5, 6))
plt.axvline(q[0], color='darkblue', linewidth=2)
plt.axvline(q[1], color='darkblue', linewidth=2)

plt.plot(x2, y2, 'k-', lw=2)
plt.show()
