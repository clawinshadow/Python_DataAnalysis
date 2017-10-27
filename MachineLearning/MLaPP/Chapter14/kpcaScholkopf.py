import numpy as np
import scipy.io as sio
import scipy.linalg as sl
import sklearn.decomposition as sd
import matplotlib.pyplot as plt

# prepare data
data = sio.loadmat('kpcaScholkopf.mat')
print(data.keys())
x_train = data['patterns']

rbf_var = 0.1
N = 15 # grid interval
x_range = np.linspace(-1, 1, 15)
y_range = np.linspace(-0.5, 1.5, 15)
xx, yy = np.meshgrid(x_range, y_range)
x_test = np.c_[xx.ravel(order='F'), yy.ravel(order='F')] # keep the same as in matlab code
print(x_test.shape)

# Fit with KPCA
def GramMat(X, centroids, rbf_var):
    N, D = X.shape
    NK = len(centroids)
    gram = np.zeros((N, NK))
    for i in range(NK):
        center = centroids[i]
        norm = sl.norm(X - center, ord=2, axis=1)**2
        gram[:, i] = np.exp(-norm / rbf_var)

    return gram

K = GramMat(x_train, centroids=x_train, rbf_var=rbf_var)
# then center the gram matrix in feature space
N, D = x_train.shape
O = np.ones((N, N)) / N
K_n = K - np.dot(O, K) - np.dot(K, O) + np.dot(O, np.dot(K, O))  # N * N

U, V = sl.eig(K_n)
U2 = np.sqrt(np.real(U))
V = V / U2  # V 本来是normalized,但是这一变化就不再是normalize的了，但是X.T*V将会变成normalize的
V[V == np.nan] = 0
print(V)

K_test = GramMat(x_test, x_train, rbf_var)  # 注意这里的centroids依然用的是训练集的点集，所以shape = N_test * N
# center in feature space
N_test, D = x_test.shape
O_test = np.ones((N_test, N)) / N
# pay attention to the different O & K, there is a mistake in Algorithm in the book
K_test_n = K_test - np.dot(O_test, K) - np.dot(K_test, O) + np.dot(O_test, np.dot(K, O))  # N_test * N
test_features = np.zeros((N_test, 8))
test_features = np.dot(K_test_n, V[:, :8])
print(test_features)

# Fit with sklearn
# 一定要注意sklearn中每个kernel参数的定义跟我们的示例中一般式不一样的，要参考以下sklearn的文档
kpca = sd.KernelPCA(N, kernel='rbf', gamma=1/rbf_var).fit(x_train)
print('eigen values by sklearn.kpca: ', kpca.lambdas_)
print('eigen vectors by sklearn.kpca: ', kpca.alphas_)  # 这个是未经变化过的原始特征向量
kpca_x_test = kpca.transform(x_test)
print('test features by sklearn: ', kpca_x_test)  # the same as test_feature

# plots
fig = plt.figure(figsize=(12, 7))
fig.canvas.set_window_title('kpcaScholkopf')

def plot(i, eigval, vec):
    subIndex = (int)('24' + str(i + 1))
    ax = plt.subplot(subIndex)
    plt.title('Eigenvalue={0:.5}'.format(eigval), fontdict={'fontsize': 15})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.axis([-1, 1, -0.5, 1.5])
    plt.xticks(np.linspace(-1, 1, 3))
    plt.yticks(np.linspace(-0.5, 1.5, 5))
    plt.plot(x_train[:, 0], x_train[:, 1], 'ro', ms=2)
    # plt.contour 与 matlab中的contour取levels的方法不太一样，所以图形不一样，只要数据是一样的就可以了
    plt.contour(xx, yy, vec.reshape(xx.shape, order='F'), 9, colors='midnightblue')

for i in range(8):
    plot(i, U[i], test_features[:, i])

plt.tight_layout()
plt.show()