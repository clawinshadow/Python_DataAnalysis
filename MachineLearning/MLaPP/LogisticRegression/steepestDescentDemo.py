import numpy as np
import scipy.linalg as sl
import scipy.optimize as so
import matplotlib.pyplot as plt

def func(x):
    return 0.5 * (x[0]**2 - x[1])**2 + 0.5 * (x[0] - 1)**2

def Jac(x):
    # 所以一阶导数函数就是一个 D 维的数组了，不再是一个标量
    differential_x0 = 2 * x[0] * (x[0]**2 - x[1]) + (x[0] - 1)
    differential_x1 = x[1] - x[0]**2

    return np.array([differential_x0, differential_x1])

def CalcZ(X, Y):
    if X.shape != Y.shape:
        raise ValueError('X and Y must have same shape')
    x = X.ravel()
    y = Y.ravel()
    z = []
    for i in range(len(x)):
        z.append(func([x[i], y[i]]))

    return np.array(z).reshape(X.shape)

x0 = [0, 0]
res = so.minimize(func, x0)
x_solution = res.x
print('solution by so.minimize(): ', res)

X, Y = np.meshgrid(np.linspace(0, 2, 200), np.linspace(-0.5, 3, 200))
# 计算Z，除了各种多元统计模型的pdf之外，我们自己写的函数一般只有先ravel再reshape比较容易计算Z值
Z = CalcZ(X, Y)

fig = plt.figure(figsize=(11, 5))
fig.canvas.set_window_title('steepestDescentDemo')

def Draw(index, eta, maxIter=20):
    i = 0
    x0 = np.array([0, 0])
    points = x0
    while i < maxIter:
        g = Jac(x0)
        x_next = x0 - eta * g
        points = np.vstack((points, x_next))
        x0 = x_next
        i += 1

    plt.subplot(index)
    plt.axis([0, 2, -0.5, 3])
    plt.xticks(np.linspace(0, 2, 5))
    plt.yticks(np.arange(-0.5, 3.1, 0.5))
    plt.contour(X, Y, Z, 60, cmap='jet')
    plt.plot(x_solution[0], x_solution[1], color='r', ls='none', marker='o', ms=5)
    plt.plot(points[:, 0], points[:, 1], color='r', marker='o', fillstyle='none', ms=3, lw=0.5)

Draw(121, 0.1)
Draw(122, 0.6)
plt.show()
