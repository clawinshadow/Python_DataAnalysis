import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))

def bohning_bound(ksi, eta):
    a = 1/4
    b = a * ksi - sigmoid(ksi)
    c = 0.5 * a * ksi**2 - sigmoid(ksi) * ksi + np.log(1 + np.exp(ksi))
    bound = 0.5 * a * eta**2 + b * eta + c

    return np.exp(-bound)

def Lambda(ksi):
    return (sigmoid(ksi) - 0.5) / (2 * ksi)

def JJ_bound(ksi, eta):
    a = 2 * Lambda(ksi)
    b = -0.5
    c = -Lambda(ksi) * ksi**2 - 0.5 * ksi + np.log(1 + np.exp(ksi))
    bound = 0.5 * a * eta**2 + b * eta + c

    return np.exp(-bound)

ksi = -2.5
xs = np.linspace(-6, 6, 200)
y_sigmoid = sigmoid(xs)
y_bohning = bohning_bound(ksi, xs)
lines_bohning = np.array([[2.5, bohning_bound(ksi, 2.5)]])

ksi = 2.5
y_jj = JJ_bound(ksi, xs)
lines_jj = np.array([[-2.5, JJ_bound(ksi, -2.5)],
                     [2.5, JJ_bound(ksi, 2.5)]])

# plots
fig = plt.figure(figsize=(10.5, 5))
fig.canvas.set_window_title('sigmoidLowerBounds')

def plot(index, title, y, lines):
    ax = plt.subplot(index)
    ax.tick_params(direction='in')
    plt.title(title)
    plt.axis([-6, 6, 0, 1])
    plt.xticks(np.linspace(-6, 6, 7))
    plt.yticks(np.linspace(0, 1, 11))
    plt.plot(xs, y_sigmoid, 'r-')
    plt.plot(xs, y, 'b:')
    for co in lines:
        plt.plot([co[0], co[0]], [0, co[1]], 'g-')

plot(121, r'$Bohning bound, \chi = -2.5$', y_bohning, lines_bohning)
plot(122, r'$JJ bound, \chi = 2.5$', y_jj, lines_jj)

plt.tight_layout()
plt.show()