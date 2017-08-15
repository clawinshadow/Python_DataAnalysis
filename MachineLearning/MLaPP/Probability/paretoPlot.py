import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

'''
scipy.stats 里面的pareto分布的frozen里面参数只有一个b，相当于书里面的k，m则是scale
'''

x = np.linspace(0, 5, 200)

k1, m1 = 0.1, 0.01
k2, m2 = 0.5, 0.001
k3, m3 = 1.0, 1.0

plt.figure(figsize=(11, 5))
plt.subplot(121)
plt.plot(x, ss.pareto.pdf(x, k1, 0, m1), 'b-', linewidth=2, label='m=0.01, k=0.10')
plt.plot(x, ss.pareto.pdf(x, k2, 0, m2), 'r:', linewidth=2, label='m=0.001, k=0.50')
plt.plot(x, ss.pareto.pdf(x, k3, 0, m3), 'k-.', linewidth=2, label='m=1.00, k=1.00')
plt.legend()
plt.ylim(-0.02, 2.5)
plt.xlim(0, 5)
plt.title('Pareto Distribution')

# 注意使用plt.loglog方法来画x和y轴都是对数单位的图形，相似的API还有semilogy和semilogx
plt.subplot(122)
m = 1.0
x2 = np.linspace(1, 5, 200)
plt.loglog(x2, ss.pareto.pdf(x2, 1, 0, m), 'b-', linewidth=2, label='k=1.0')
plt.loglog(x2, ss.pareto.pdf(x2, 2, 0, m), 'r:', linewidth=2, label='k=2.0')
plt.loglog(x2, ss.pareto.pdf(x2, 3, 0, m), 'k-.', linewidth=2, label='k=3.0')
plt.title('Pareto(m=1, k) on log scale')
plt.xlim(1, 5)
plt.legend()

plt.show()
