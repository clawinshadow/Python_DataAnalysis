import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

N = 100000    # number of samples
samples = np.random.randn(N)
x = np.linspace(-4, 4, 801)
cdf = np.zeros(x.shape)
u = 0
for i in range(len(x)):
    # samples all sampled from N(0, 1), then we can calculate the prob of (samples < xi) to approximate cdf of N(0, 1)
    cdf[i] = np.sum(samples < x[i]) / N
    if cdf[i] > 0.5 and u == 0:
        u = i

# plot
fig = plt.figure()
fig.canvas.set_window_title('sampleCdf')

ax = plt.subplot()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.axis([-4, 4.5, 0, 1.1])
plt.xticks([x.min(), x[u]], ['0', 'x'], size=15, weight='bold')
plt.yticks([0, cdf[u], 1], ['0', 'u', '1'], size=15, weight='bold')
plt.plot(x, cdf, '-', lw=2)
plt.plot([x.min(), x.max()], [1, 1], 'g--', lw=2)
plt.text(4.1, 1, 'F', size=15, weight='bold')
plt.arrow(x.min(), cdf[u], x[u] - x.min(), 0, length_includes_head=True,\
          width=0.01, color='r', edgecolor='b', head_length=0.2)
plt.arrow(x[u], cdf[u], 0, -cdf[u], length_includes_head=True,\
          width=0.05, color='r', edgecolor='b', head_length=0.04)

plt.tight_layout()
plt.show()