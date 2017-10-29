import numpy as np
import matplotlib.pyplot as plt

def GaussianKernel(x):
    val1 = 1 / np.sqrt(2 * np.pi)
    val2 = np.exp(-x**2 / 2)

    return val1 * val2

def EpanechnikovKernel(x):
    indices = np.abs(x) <= 1
    return 3/4 * (1 - x**2) * indices

def TricubeKernel(x):
    indices = np.abs(x) <= 1
    val = 70/81 * (1 - np.abs(x)**3)**3

    return val * indices

def BoxcarKernel(x):
    return np.abs(x) <= 1

x = np.linspace(-1.5, 1.5, 301)
y1 = 0.5 * BoxcarKernel(x)
y2 = EpanechnikovKernel(x)
y3 = TricubeKernel(x)
y4 = GaussianKernel(x)

fig = plt.figure()
fig.canvas.set_window_title('smoothingKernelPlot')

ax = plt.subplot()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.axis([-1.5, 1.5, 0, 0.9])
plt.xticks(np.linspace(-1.5, 1.5, 7))
plt.yticks(np.linspace(0, 0.9, 10))
plt.plot(x, y1, color='midnightblue', lw=2, label='Boxcar')
plt.plot(x, y2, 'r:', lw=2, label='Epanechnikov')
plt.plot(x, y3, 'k-.', lw=2, label='Tricube')
plt.plot(x, y4, 'g--', lw=2, label='Gaussian')

plt.tight_layout()
plt.legend()
plt.show()