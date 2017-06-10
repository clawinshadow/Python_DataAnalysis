import numpy as np
import matplotlib.pyplot as plt

'''
核方法的主要思想基于一个假设："在低维空间中不能线性分割的点集，通过某个映射投影到高维空间后，很有可能变成线性可分的"

这里还有一个问题：“为什么我们要关心向量的内积？”，
一般地，我们可以把分类（或者回归）的问题分为两类：参数学习的形式和基于实例的学习形式。

1. 参数学习的形式就是通过一堆训练数据，把相应模型的参数给学习出来，然后训练数据就没有用了，
   对于新的数据，用学习出来的参数即可以得到相应的结论；

2. 而基于实例的学习（又叫基于内存的学习）instance-based or memory-based, 则是在预测的时候也会使用训练数据，如KNN算法。
   而基于实例的学习一般就需要判定两个点之间的相似程度，一般就通过向量的内积来表达。
   从这里可以看出，核方法不是万能的，它一般只针对基于实例的学习。

下面给核函数一个正式定义：设Χ为输入空间，Ω为特征空间，如果存在一个 X 到 Ω 的映射 φ(x): X → Ω
对所有的 x,z∈X，函数 κ(x,z) 满足 κ(x,z)=<φ(x), φ(z)> ,则称 φ(x) 为输入空间到特征空间的映射函数，κ(x,z) 为核函数。

所以核函数的神奇之处就在于，当我们用一个具体的映射函数将数据从低维空间投射到高维空间后，我们不用去逐个计算每个点集在
高维空间中的坐标，然后再通过高维坐标来计算内积或者采用线性判别等方法来寻找分割的超平面，而是可以将低维的坐标直接带入
核函数中计算，得出来的结果与φ(x)投射后通过高维坐标计算内积得出来的结果一样。

那么如何判定一个核函数的正确性呢，可以根据Mercer定理：

    任何半正定的函数都可以作为核函数。所谓半正定的函数f(xi,xj)，是指拥有训练数据集合（x1,x2,...xn)，
    我们定义一个矩阵的元素aij = f(xi,xj)，这个矩阵式n*n的，如果这个矩阵是半正定的，那么f(xi,xj)就称为半正定的函数。
    这个mercer定理不是核函数必要条件，只是一个充分条件，即还有不满足mercer定理的函数也可以是核函数。
    常见的核函数有高斯核，多项式核等等。

    通过这些basis核函数，可以通过如下变化得出新的核函数：
    给定合法的核函数k1(x, z)和k2(x, z)，那么以下核函数都是合法的：
        k(x, z) = c*k1(x, z)               # c是常数
        k(x, z) = f(x)*k1(x, z)f(z)        # f(x)是任意函数
        k(x, z) = q(k1(x, z))              # q(x)是任意多项式，所有系数不为负
        k(x, z) = exp(k1(x, z))           
        k(x, z) = k1(x, z) + k2(x, z)   
        k(x, z) = k1(x, z) * k2(x, z)
        k(x, z) = k3(φ(x), φ(z))           # φ(x) 是 x到M维R空间的映射
        k(x, z) = x.t*A*x                  # A是对称正定矩阵
        k(x, z) = ka(xa, za) + kb(xb, zb) 
        k(x, z) = ka(xa, za) * kb(xb, zb)
'''

def expo(x, order):
    return np.power(x, order)

# φ(x) => (x, x2, x3, ..., x10)
def basis_polynomial(x, maxOrder=10):
    mapping = []
    for i in range(1, maxOrder + 1, 1):
        mapping.append(expo(x, i))

    return np.array(mapping)

def gaussian(x, local, scale):
    return np.exp(np.divide(np.multiply(-1, np.power(np.subtract(x, local), 2)), 2*scale**2))

def basis_gaussian(x, params):
    mapping = []
    for i in range(len(params)):
        local, scale = params[i]
        mapping.append(gaussian(x, local, scale))

    return np.array(mapping)

def sigmoidal(x, local, scale):
    return np.divide(1, np.add(1, np.exp(np.divide(np.subtract(local, x), scale))))

def basis_sigmoidal(x, params):
    mapping = []
    for i in range(len(params)):
        local, scale = params[i]
        mapping.append(sigmoidal(x, local, scale))

    return np.array(mapping)

# k(x, x_base) = <φ(x), φ(x_base)>
def kernel_method(func, x, x_base, maxOrder=10, gsParams=[(0, 0.2)]):
    if func is basis_polynomial:
        a = func(x, maxOrder).T
        b = func(x_base, maxOrder).reshape(-1, 1)
        return np.dot(a, b)
    elif func is basis_gaussian:
        a = func(x, gsParams).T
        b = func(x_base, gsParams).reshape(-1, 1)
        return np.dot(a, b)
    elif func is basis_sigmoidal:
        a = func(x, gsParams).T
        b = func(x_base, gsParams).reshape(-1, 1)
        return np.dot(a, b)
    else:
        return None 

# draw basis functions - polynomial
fig = plt.figure(figsize=(9, 6))
fig.canvas.set_window_title('Kernel Methods')    # set window title
plt.subplot(231)
count = 10
colors = ['green', 'darkblue', 'yellow', 'red', 'tan', 'olivedrab', 'coral', 'darkviolet', 'black', 'cyan', 'pink']
x = np.linspace(-1, 1, 500)
y_vals = basis_polynomial(x, count)
for i in range(len(y_vals)):
    plt.plot(x, y_vals[i], c=colors[i], lw=0.5)
plt.axis([-1, 1, -1, 1])

# draw kernel function for polynomial basis functions
x_base = -0.5  # x' = -0.5
kernel_polynomials = kernel_method(basis_polynomial, x, x_base)
print(kernel_polynomials.shape)
plt.subplot(234)
plt.plot(x, kernel_polynomials.ravel(), c='darkblue')
plt.plot(x, np.tile(0, 500), color='green', linestyle='--', linewidth=0.5)
plt.plot(x_base, -0.4, color='red', marker="x")
plt.axis([-1, 1, -0.4, 1])

# draw basis functions - gaussian
params=[(-1, 0.2),
        (-0.8, 0.2),
        (-0.6, 0.2),
        (-0.4, 0.2),
        (-0.2, 0.2),
        (0, 0.2),
        (0.2, 0.2),
        (0.4, 0.2),
        (0.6, 0.2),
        (0.8, 0.2),
        (1, 0.2)]  # 共11个gaussian
plt.subplot(232)
for i in range(len(params)):
    local, scale = params[i]
    plt.plot(x, gaussian(x, local, scale), c=colors[i], lw=0.5)

# draw kernel methods - gaussian
x_base_gaussian = 0
kernel_gaussian = kernel_method(basis_gaussian, x, x_base_gaussian, gsParams=params)
print(kernel_gaussian.shape)
plt.subplot(235)
plt.plot(x, kernel_gaussian, c='darkblue')
plt.plot(x_base_gaussian, 0, color='red', marker="x")
plt.axis([-1, 1, 0, 2])


# draw basis functions - sigmoid
params=[(-1, 0.1),
        (-0.8, 0.1),
        (-0.6, 0.1),
        (-0.4, 0.1),
        (-0.2, 0.1),
        (0, 0.1),
        (0.2, 0.1),
        (0.4, 0.1),
        (0.6, 0.1),
        (0.8, 0.1),
        (1, 0.1)]  # 共11个
plt.subplot(233)
for i in range(len(params)):
    local, scale = params[i]
    plt.plot(x, sigmoidal(x, local, scale), c=colors[i], lw=0.5)

# draw kernel methods - sigmoid
x_base_sigmoid = 0
kernel_sigmoid = kernel_method(basis_sigmoidal, x, x_base_sigmoid, gsParams=params)
print(kernel_sigmoid.shape)
plt.subplot(236)
plt.plot(x, kernel_sigmoid, c='darkblue')
plt.plot(x_base_sigmoid, 0, color='red', marker="x")
plt.axis([-1, 1, 0, 6])

plt.show()

