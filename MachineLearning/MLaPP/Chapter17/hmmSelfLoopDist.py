import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

def get_probs(d, n, p=0.99):
    log_p = np.zeros(len(d))
    for i in range(len(d)):
        di = d[i]
        if di < n:
            log_p[i] = -np.inf
        else:
            log_p[i] = np.log(sp.comb(di-1, n-1)) + (di - n) * np.log(p) + n * np.log(1 - p)

    return np.exp(log_p)

xs = np.linspace(1, 600, 600)
ns = [1, 2, 5]
ys = []
for i in range(len(ns)):
    ys.append(get_probs(xs, ns[i]))

# plots
fig = plt.figure()
fig.canvas.set_window_title('hmmSelfLoopDist')

plt.subplot()
plt.axis([0, 600, 0, 0.012])
plt.xticks(np.linspace(0, 600, 7))
plt.yticks(np.linspace(0, 0.012, 7))
plt.plot(xs, ys[0], 'b-', lw=1.5, label='n=1')
plt.plot(xs, ys[1], 'r:', lw=1.5, label='n=2')
plt.plot(xs, ys[2], 'k-.', lw=1.5, label='n=5')
plt.legend()

plt.tight_layout()
plt.show()
