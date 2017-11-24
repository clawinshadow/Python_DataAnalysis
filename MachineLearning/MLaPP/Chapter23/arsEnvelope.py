import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

'''
本示例中还是用的导数来作为斜率，切线来作为上界，其实与arsDemo中的思想还是不一样的，
arsDemo中的更为巧妙，用相邻两点之间的直线作为上界和下界，具体可参见arsDemo
'''

xs = np.linspace(-1.5, 1.5, 301)
ys = ss.norm().pdf(xs)
ys = np.log(ys)   # should be log concave

# plot
fig = plt.figure()
fig.canvas.set_window_title('arsEnvelope')

plt.subplot()
plt.plot(xs, ys, 'b-',lw=2)

def plotTangent(x1):
    # x1 = -0.7                           # 取一个样本点
    y1 = np.log(ss.norm().pdf(x1))      # 对应的y值
    x1_next = x1 + 0.05
    y1_next = ss.norm().logpdf(x1_next) # 取delta=0.05来计算数值导数
    tangent = (y1_next - y1) / 0.05     # 数值导数，x1点处切线的斜率
    plt.plot([x1 - 0.5, x1 + 0.5], [y1 - tangent * 0.5, y1 + tangent * 0.5], 'r-', lw=2)
    plt.plot([x1, x1], [ys.min(), y1], 'r--', lw=2)

plotTangent(-0.7)
plotTangent(0)
plotTangent(0.7)

# plt.axis('off')
plt.tight_layout()
plt.show()