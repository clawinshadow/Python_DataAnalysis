import numpy as np
import matplotlib.pyplot as plt

'''
p(wj|a, b) = (a/2b) * (1 + |wj| / b)**-(a+1)
画的是 -log[p(w1)] - log[p(w2)]
'''

def neglogpdf(w, a, b):
    return (a + 1) * np.log(1 + np.abs(w) / b) - np.log(a / (2 * b))

def get_nlps(w1, w2, a, b):
    w1_ravel = w1.ravel()
    w2_ravel = w2.ravel()
    nlps = neglogpdf(w1_ravel, a, b) + neglogpdf(w2_ravel, a, b)
    nlps = nlps.reshape(w1.shape)
    plot_level = nlps.min() + 0.5 * (nlps.max() - nlps.min())
    
    return nlps, plot_level

w1, w2 = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
nlps1, level1 = get_nlps(w1, w2, 1, 0.01)
nlps2, level2 = get_nlps(w1, w2, 1, 0.10)
nlps3, level3 = get_nlps(w1, w2, 1, 1.00)

fig = plt.figure()
fig.canvas.set_window_title('starfish')

ax = plt.subplot(aspect='equal')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title('HAL')
plt.axis([-1, 1, -1, 1])
plt.xticks(np.linspace(-1, 1, 11))
plt.yticks(np.linspace(-1, 1, 11))
plt.plot([0], [0], marker='o', ms=3, color='k')
plt.contour(w1, w2, nlps1, levels=[level1], colors='midnightblue')
plt.contour(w1, w2, nlps2, levels=[level2], linestyles='dotted', colors='r')
plt.contour(w1, w2, nlps3, levels=[level3], linestyles='dashdot', colors='k')

plt.legend(['1', '2', '3'])
plt.tight_layout()
plt.show()

