import numpy as np
import scipy.linalg as sl
import scipy.io as sio
import sklearn.decomposition as sd
import matplotlib.pyplot as plt

'''
TruncatedSVD 和 PCA的区别在于，TSVD的fit使用初始训练集矩阵就可以，但是PCA需要标准化X之后才能Fit，详见sklearn User Guide
'''

# load data
data = sio.loadmat('clown.mat')
print(data)
print('data.X shape: ', data['X'].shape)
clownImg = data['X']  # 200 * 320

# plot clowns
fig = plt.figure()
fig.canvas.set_window_title('svdImageDemo')

KS = [2, 5, 20]

def svdPlot(i):
    k = KS[i]
    TSVD = sd.TruncatedSVD(k)
    TSVD.fit(clownImg.T)   # N = 320, D = 200
    w = TSVD.components_   # L * D
    clown_reduce = TSVD.transform(clownImg.T)  # N * L
    clown_approx = np.dot(clown_reduce,  w).T  # D * N, 恢复到 200 * 320
    plt.subplot((int)('22' + str(i + 2)))
    plt.axis('off')
    plt.title('rank ' + str(k))
    plt.imshow(clown_approx, cmap='gray')

plt.subplot(221)
plt.axis('off')
plt.title('rank 200')
plt.imshow(clownImg, cmap='gray')
    
for i in range(len(KS)):
    svdPlot(i)

# plot singular values
u, s, vt = sl.svd(clownImg.T)
s = np.log(s[:100])
print(s.min(), s.max())

# 1. 多维数组需要转化为一维之后再shuffle才更彻底
# 2. permutation 与 shuffle 的不同之处在于，permutation不改变原来的数组，返回一个copy
clown_shuffle = np.random.permutation(clownImg.ravel())
clown_shuffle = clown_shuffle.reshape(clownImg.shape)
u2, s2, vt2 = sl.svd(clown_shuffle.T)
s2 = np.log(s2[:100])

fig2 = plt.figure()
fig2.canvas.set_window_title('svdImageDemo_2')

plt.subplot()
plt.axis([0, 100, 4, 10])
plt.plot(np.linspace(1, 100, 100), s, 'r-', label='original')
plt.plot(np.linspace(1, 100, 100), s2, 'g:', label='randomized')
plt.xlabel('i')
plt.ylabel(r'$log(\delta_i)$')
plt.legend()

plt.show()
