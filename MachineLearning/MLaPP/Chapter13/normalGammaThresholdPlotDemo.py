import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

def neglogpdf(w, a, b):
    return (a + 1) * np.log(1 + np.abs(w) / b) - np.log(a / (2 * b))

z = np.arange(-10, 10, 0.5)
x = np.arange(-10.005, 10, 0.05)
bs = [0.01, 0.1, 1]
out_laplace = np.zeros(len(z))
out_gt = np.zeros((len(bs), len(z)))
for i in range(len(z)):
    c = 1
    out_laplace[i] = np.argmin(0.5 * (z[i] - x)**2 + c * np.abs(x))

    a = 1
    for j in range(len(bs)):
        b = bs[j]
        out_gt[j, i] = np.argmin(0.5 * (z[i] - x)**2 + neglogpdf(x, a, b))
        

out_laplace = np.array(out_laplace, dtype='int32')
out_gt = np.array(out_gt, dtype='int32')

# plots
fig = plt.figure(figsize=(11, 5))
fig.canvas.set_window_title('normalGammaThresholdPlotDemo')

def plot_basis():
    plt.xlabel(r'$w^{MLE}$')
    plt.ylabel(r'$w^{MAP}$')
    plt.axis([-10, 10, -10, 10])
    plt.xticks(np.linspace(-10, 10, 6))
    plt.yticks(np.linspace(-10, 10, 6))
    plt.plot(z, z, 'r:', lw=2)

plt.subplot(121)
plt.title('Lasso')
plot_basis()
plt.plot(z, x[out_laplace], color='midnightblue', lw=2)

plt.subplot(122)
plt.title('HAL')
plot_basis()
for i in range(len(out_gt)):
    idx = out_gt[i]
    plt.plot(z, x[idx], ls=':', lw=2, label='b = {0}, a={1}'.format(bs[i], 1))

plt.legend()
plt.tight_layout()
plt.show()
