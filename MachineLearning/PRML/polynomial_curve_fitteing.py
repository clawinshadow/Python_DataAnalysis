import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as plt

'''
demos about PRML - Chapter 1.1 : Polynomial Curve Fitting 
'''

# M=0时， 零阶多项式
def func0(xdata, m0):
    return m0 * np.ones(len(xdata))

# M = 1时， 一阶多项式
def func1(xdata, m0, m1):
    return m0 + m1 * xdata

def func3(xdata, m0, m1, m2, m3):
    return m0 + m1 * xdata + m2 * xdata**2 + m3 * xdata**3

def func9(xdata, m0, m1, m2, m3, m4, m5, m6, m7, m8, m9):
    return m0 + m1 * xdata + m2 * xdata**2 + m3 * xdata**3 + m4 * xdata**4 + m5 * xdata**5 + m6 * xdata**6 +\
           m7 * xdata**7 + m8 * xdata**8 + m9 * xdata**9

x = np.linspace(0, 1, 10)
x_dense = np.linspace(0, 1, 200)      # to get a smooth curve, need more data points
y_dense = np.sin(2 * np.pi * x_dense) # correspond to x_dense
perfect_y = np.sin(2 * np.pi * x)
noise = np.random.randn(len(x)) * 0.2
real_y = perfect_y + noise

popt, pcov = so.curve_fit(func0, x, real_y)         # popt是对应的系数的最小二乘估计, 是长度为参数数量的一维数组 
print('curve fit parameters when M = 0: ', popt)
fit_y_0 = func0(x_dense, popt[0])                   # 用x_dense是为了画一条更平滑的曲线, M = 0时

popt, pcov = so.curve_fit(func1, x, real_y)         
print('curve fit parameters when M = 1: ', popt)
fit_y_1 = func1(x_dense, popt[0], popt[1])          # M = 1 时

popt, pcov = so.curve_fit(func3, x, real_y)         
print('curve fit parameters when M = 3: ', popt)
fit_y_3 = func3(x_dense, popt[0], popt[1], popt[2], popt[3]) # M = 3 时

popt, pcov = so.curve_fit(func9, x, real_y)         
print('curve fit parameters when M = 9: ', popt)
fit_y_9 = func9(x_dense, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9])  # M = 9 时

print('data: ', x)
print('target: ', real_y)

plt.figure(figsize=(11, 7))           # figsize的单位是inch, 不太好估量
ax = plt.subplot(221)
plt.scatter(x, real_y, c='white', marker='o', edgecolors='blue', label='Training Data')
plt.plot(x_dense, y_dense, c='green', label='Real Curve')
plt.plot(x_dense, fit_y_0, c='red', label='Fit Curve')
plt.xlabel('x')
plt.ylabel('y')
# 文本的位置按x和y轴的比例来决定
plt.text(0.45, 0.9, 'M = 0', color='burlywood', fontsize=12, transform=ax.transAxes)
plt.legend()                          # 调用该方法才会展示label参数中指定的图例名

# M = 1
ax2 = plt.subplot(222)
plt.scatter(x, real_y, c='white', marker='o', edgecolors='blue', label='Training Data')
plt.plot(x_dense, y_dense, c='green', label='Real Curve')
plt.plot(x_dense, fit_y_1, c='red', label='Fit Curve')
plt.xlabel('x')
plt.ylabel('y')
plt.text(0.45, 0.9, 'M = 1', color='burlywood', fontsize=12, transform=ax2.transAxes)                                              

# M = 3
ax3 = plt.subplot(223)
plt.scatter(x, real_y, c='white', marker='o', edgecolors='blue', label='Training Data')
plt.plot(x_dense, y_dense, c='green', label='Real Curve')
plt.plot(x_dense, fit_y_3, c='red', label='Fit Curve')
plt.xlabel('x')
plt.ylabel('y')
plt.text(0.45, 0.9, 'M = 3', color='burlywood', fontsize=12, transform=ax3.transAxes)                                             

# M = 9
ax4 = plt.subplot(224)
plt.scatter(x, real_y, c='white', marker='o', edgecolors='blue', label='Training Data')
plt.plot(x_dense, y_dense, c='green', label='Real Curve')
plt.plot(x_dense, fit_y_9, c='red', label='Fit Curve')
plt.xlabel('x')
plt.ylabel('y')
plt.text(0.45, 0.9, 'M = 9', color='burlywood', fontsize=12, transform=ax4.transAxes)

fig = plt.gcf()
fig.canvas.set_window_title('Polynomial Curve Fitting ')    # 设置window的名字
plt.show()
