import numpy as np
import scipy.io as sio
import sklearn.preprocessing as sp
import matplotlib.pyplot as plt
from pcaFit import *

# extract data from MNIST
data = sio.loadmat('mnistAll.mat')
# print(data['mnist'])        # 是个list
data = data['mnist'][0, 0]    # shape 是个[1, 1], 里面是个tuple，包含四个ndarray
print('train images shape: ', data[0].shape)  # train images, 28 * 28 * 60000, 即60000个 28*28的灰度图象
print('test images shape: ', data[1].shape)   # test images, 28 * 28 * 10000, 10000个 28*28 的灰度图象作为测试集
print('train image labels shape: ', data[2].shape)
print('test image labels shape: ', data[3].shape)

N = 1000                  # 只选1000个为 3 的训练集
train_all = data[0]
train_all_labels = data[2]
h, w, n = train_all.shape # 28 * 28 * 60000

indices_3 = train_all_labels == 3
train_3 = train_all[:, :, indices_3.ravel()] # 筛选出所有为 3 的图像
print(train_3.shape)
X = train_3[:, :, 0:N]
print(X.shape)
X = X.reshape((h*w, N))   # 将28 * 28的像素压缩为一列，共1000列, ()
print(X.shape)

# fit pca
mu = np.mean(X, axis=1)
rank = np.linalg.matrix_rank(X)
mu2, V, vr, z, x_recon = PCA(X.T, rank)
scaler = spp.MinMaxScaler(feature_range=(0, 255)) # 因为特征向量被nomarlize过了，重新scale到[0, 255]的色域范围内
vr2 = np.copy(vr)
vr2[:, 0] *= -1                                    # scipy中计算特征向量很容易把符号翻转，对比书中的图像，这里也需要翻转过来
vr2 = scaler.fit_transform(vr) 
print('V: \n', vr2)
print(vr2.shape)

# plots
fig1 = plt.figure()
fig1.canvas.set_window_title('pcaImageDemo')

# mean
plt.subplot(221)
plt.axis('off')
plt.title('mean')
plt.imshow(mu.reshape((h, w)), cmap='gray')

# principal component 
plt.subplot(222)
plt.axis('off')
plt.title('principal basis 1')
plt.imshow(vr2[:, 0].reshape((h, w)), cmap='gray')

plt.subplot(223)
plt.axis('off')
plt.title('principal basis 2')
plt.imshow(vr2[:, 1].reshape((h, w)), cmap='gray')

plt.subplot(224)
plt.axis('off')
plt.title('principal basis 3')
plt.imshow(vr2[:, 2].reshape((h, w)), cmap='gray')

# reconstruct
def plotRecon(i, L):
    mu3, V, vr3, z3, x_recon3 = PCA(X.T, L)
    plt.subplot((int)('22' + str(i + 1)))
    plt.axis('off')
    plt.title('reconstructed with {0} bases'.format(L))
    plt.imshow(x_recon3[125].astype('float64').reshape((h, w)), cmap='gray')  # 任意挑选一个，这里选第125个

fig2 = plt.figure()
fig2.canvas.set_window_title('pcaImageDemo_recon')

LS = [2, 10, 100, 506]
for i in range(len(LS)):
    plotRecon(i, LS[i])

plt.show()
