import numpy as np
import sklearn.preprocessing as sp
import matplotlib.pyplot as plt
from mixexpFit import *

'''需要引入同级目录中的mixexpFit.py'''

def GetColors(X, V):
    colors = []
    for i in range(len(X)):
        probs = Softmax(X[i], V)
        color = [probs[1], probs[2], probs[0]]  # 按r, g, b来排列
        colors.append(color)

    # print(np.array(colors))
    return colors

np.random.seed(0)

# generate data
n = 200
y = np.random.rand(n)
noise = 0.05 * np.random.randn(n)
x = y + 0.3 * np.sin(2 * np.pi * y) + noise

# plot datas
fig = plt.figure(figsize=(11, 10))
fig.canvas.set_window_title("mixexpDemoOneToMany")

plt.subplot(221)
plt.axis([0, 1, -0.2, 1.2])
plt.xticks(np.arange(0, 1.1, 0.2))
plt.yticks(np.arange(-0.2, 1.3, 0.2))
plt.title('forwards problem')
plt.plot(y, x, 'bo', fillstyle='none')

# Fit inverse problem
K = 3
x_train = x.reshape(-1, 1)
SS = sp.StandardScaler()
SS.fit(x_train)    # 记住参数
x_train = SS.transform(x_train)
x_train = np.c_[np.ones(len(x_train)).reshape(-1, 1), x_train]
y_train = y.reshape(-1, 1)

V, W, sigmas = FitMixExp(x_train, y_train, K, 60)
# V = np.array([[1.15695073, -0.14377757], [-0.88400753, 2.23956147], [-0.2729432, -2.09578391]])
# W = np.array([[0.5264722, -0.22944983], [0.80251986, 0.0719835], [0.20709093, 0.08431157]])
# sigmas = np.array([ 0.08459333, 0.01885704, 0.02406214]) 

# plots subplot 2
xx = np.linspace(0, 1, 200)
xx_standard = SS.transform(xx.reshape(-1, 1))
xx_standard = np.c_[np.ones(len(xx_standard)).reshape(-1, 1), xx_standard]
yy = np.dot(xx_standard, W.T)

plt.subplot(222)
plt.title('expert predictions')
plt.axis([-0.2, 1.2, -0.2, 1.2])
plt.xticks(np.arange(-0.2, 1.3, 0.2))
plt.yticks(np.arange(-0.2, 1.3, 0.2))
plt.plot(xx, yy[:, 0], 'b-')
plt.plot(xx, yy[:, 1], 'r-')
plt.plot(xx, yy[:, 2], 'g-')

colors = GetColors(x_train, V)  # 根据概率之比来混合不同的颜色
plt.scatter(x, y, c=colors, marker='D', s=5)

# Predict
# generate test set
sorted_x = np.sort(x)
indices = np.arange(0, len(sorted_x), 4)   # 每隔4个采集一个样本点作为测试集
x_test = sorted_x[indices]
x_test_standard = SS.transform(x_test.reshape(-1, 1))
x_test_standard = np.c_[np.ones(len(x_test_standard)).reshape(-1, 1), x_test_standard.reshape(-1, 1)]

# calculate predictions
muk = np.dot(x_test_standard, W.T)         # 每一行表示分属于三个专家的均值
mus = []
modes = []
for i in range(len(x_test_standard)):
    pi = Softmax(x_test_standard[i], V).ravel()
    maxindex = np.argsort(pi)[-1]          # 预测的众数就是取概率最大的那个专家的均值
    modes.append(muk[i, maxindex])
    muki = muk[i].ravel()
    mui = np.sum(muki * pi)
    mus.append(mui)                        # 预测的均值是对三个专家的均值的加权平均，很多时候这种加权平均并不一定是好的预测

# plot subplot 3
plt.subplot(223)
plt.title('prediction')
plt.axis([-0.2, 1.2, -0.1, 1.1])
plt.xticks(np.arange(-0.2, 1.3, 0.2))
plt.yticks(np.arange(0, 1.01, 0.1))
plt.plot(x, y, 'bo', fillstyle='none')
plt.plot(x_test, mus, 'rx', label='mean')
plt.plot(x_test, modes, 'ks', fillstyle='none', label='mode')
plt.legend()

plt.show()
