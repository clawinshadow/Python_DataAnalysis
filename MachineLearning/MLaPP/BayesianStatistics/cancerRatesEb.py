import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import scipy.special as sp

'''
使用到EB方法的一般比较难求解，没有固定的公式，多数都是要用数值最优化的方法来求解
'''

def di_pochhammer(x, n):
    if n.ndim == 2:
        index = n > 0
        y = np.zeros(n.shape)
        y[index] = sp.digamma(x[index] + n[index]) - sp.digamma(x[index])
    else:
        y = sp.digamma(x + n) - sp.digamma(x)

    return y

def dirichlet_moment_match(p):
    a = np.mean(p, axis=0)
    m2 = np.mean(np.dot(p.T, p), axis=0)
    s = (a - m2) / (m2 - a**2)
    s = np.median(s)
    if s == 0:
        s = 1
    return a * s

def polya_moment_match(data):
    sdata = np.sum(data, axis=1).reshape(-1, 1)
    p = data / np.tile(sdata, data.shape[1])
    return dirichlet_moment_match(p)

def polya_fit_simple(data):
    a = polya_moment_match(data)
    N, K  = data.shape
    sdata = np.sum(data, axis=1)
    for i in range(6000):
        old_a = a
        sa = sum(a)
        t = np.tile(a, N).reshape(data.shape)
        g = np.sum(di_pochhammer(t, data), axis=0)
        h = np.sum(di_pochhammer(sa, sdata))
        a = a * g / h
        if np.allclose(a, old_a):
            break

    return a

y = np.array([0, 0, 2, 0, 1, 1, 0, 2, 1, 3, 0, 1, 1, 1, 54, 0, 0, 1, 3, 0])
n = np.array([1083, 855, 3461, 657, 1208, 1025, 527, 1668, 583, 582, 917, 857, 680, \
              917, 53637, 874, 395, 581, 588, 383])

x = np.c_[y.reshape(-1, 1), (n - y).reshape(-1, 1)]
prior = polya_fit_simple(x)
print('prior hyperparamters: ', prior)

a, b = prior[0], prior[1]
MLE = (y / n) * 1000
pooledMLE = np.sum(y) * 1000 / np.sum(n)
aPost = a + y
bPost = b + n - y
postMean = aPost * 1000 / (aPost + bPost)
popMean = a * 1000 / (a + b)
lefts = np.linspace(1, 20, 20)

postMedian = np.zeros(len(y))
intervals = []
for i in range(len(aPost)):
    ai, bi = aPost[i], bPost[i]
    post = ss.beta(ai, bi)
    postMedian[i] = post.median()
    intervals.append(list(post.interval(0.95)))

postMedian = postMedian * 1000
intervals = np.array(intervals).T * 1000 # x scale is 10e-3
intervals[0] = postMedian - intervals[0]
intervals[1] = intervals[1] - postMedian
print('postMedian: ', postMedian)
print('intervals: \n', intervals)

fig1 = plt.figure(figsize=(8, 7))
fig1.canvas.set_window_title('cancerRatesEb_1')

plt.subplot(411)
plt.axis([0, 25, 0, 5])
plt.xticks(np.linspace(0, 25, 6))
plt.yticks([0, 5])
plt.title('number of people with cancer (truncated at 5)')
plt.bar(lefts, y, color='darkblue', edgecolor='k', linewidth=0.5, align='center')

plt.subplot(412)
plt.title('pop of city (truncated at 2000)')
plt.axis([0, 25, 0, 2000])
plt.xticks(np.linspace(0, 25, 6))
plt.yticks([0, 1000, 2000])
plt.bar(lefts, n, color='darkblue', edgecolor='k', linewidth=0.5, align='center')

plt.subplot(413)
plt.title('MLE*1000 (red line=pooled MLE)')
plt.axis([0, 25, 0, 10])
plt.xticks(np.linspace(0, 25, 6))
plt.yticks([0, 5, 10])
plt.bar(lefts, MLE, color='darkblue', edgecolor='k', linewidth=0.5, align='center')
plt.hlines(pooledMLE, 0, 20, colors='r', lw=2)

plt.subplot(414)
plt.title('posterior mean*1000 (red line=pop mean)')
plt.axis([0, 25, 0, 4])
plt.xticks(np.linspace(0, 25, 6))
plt.yticks([0, 2, 4])
plt.bar(lefts, postMean, color='darkblue', edgecolor='k', linewidth=0.5, align='center')
plt.hlines(popMean, 0, 20, colors='r', lw=2)

fig1.subplots_adjust(hspace=0.7)

fig2 = plt.figure(figsize=(6, 5))
fig2.canvas.set_window_title('cancerRatesEb_2')

plt.subplot()
plt.title('95% credible interval on theta, *=median')
plt.axis([0, 8, 0, 21])
plt.xticks(np.linspace(0, 8, 9))
plt.yticks(np.linspace(0, 20, 11))
plt.errorbar(postMedian, lefts, xerr=intervals, marker='*', color='darkblue',\
             mew=0.5, lw=0.5, linestyle='none')
plt.xlabel(r'$10^{-3}$')

plt.show()


