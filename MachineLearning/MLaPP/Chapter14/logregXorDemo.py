import numpy as np
import scipy.stats as ss
import scipy.linalg as sl
import sklearn.linear_model as slm
import sklearn.preprocessing as sp
import matplotlib.pyplot as plt

'''
与书中的图形无法保持一致，因为原始数据的一点点不一样就会造成决策边界的完全不同。
拿不到书中的原始数据，所以画不出来一样的图形。
'''

np.random.seed(0)

# generate data
def poly(X, deg=10):
    '''
    这个是多元的X的指数扩展，与之前一维的X是不一样的，与sklearn里面的PolynomialFeatures也不一样
    以如下X为例，deg=2
       1  0            1  0  1  0
    X: 2  4   -> φ(X): 2  4  4  16
       3  5            3  5  9  25 
    '''
    N, D = X.shape
    result = np.zeros((N, D * deg))
    for i in range(deg):
        result[:, i*D : (i+1)*D] = X**(i + 1)

    return result
    
mus = np.array([[1, 1], [5, 5], [1, 5], [5, 1]])
sigma = 0.5 * np.eye(2)
N = 20
X = np.zeros((4 * N, 2))
y = np.zeros(4 * N)
for i in range(len(mus)):
    mu = mus[i]
    X[i * N : (i + 1) * N] = ss.multivariate_normal(mu, sigma).rvs(N)
y[2*N:] = 1

# Fit with poly 10 expaned logistic regression
x_expanded = poly(X) # polynomial expansion with deg=10
print(x_expanded.shape)
scaler = sp.MinMaxScaler(feature_range=(-1, 1)).fit(x_expanded) # because number is too large, so rescale to (-1, 1)
X_train = scaler.transform(x_expanded) 
LG = slm.LogisticRegression(C=50).fit(X_train, y)

# for plot decision boundary
xx, yy = np.meshgrid(np.linspace(-1, 7, 200), np.linspace(-1, 7, 200))
xx_ravel = xx.ravel()
yy_ravel = yy.ravel()
xgrid_expanded = poly(np.c_[xx_ravel, yy_ravel])
xgrid_rescale = scaler.transform(xgrid_expanded)

Z = LG.predict(xgrid_rescale)
print('distinct Z: ', np.unique(Z))
Z = Z.reshape(xx.shape)
y_predict = LG.predict(X_train)
print('error rate: ', 1 - np.count_nonzero(y == y_predict) / len(y))

# Fit with 4-centroids kernelised model with scale=1 RBF kernel
def kernelise(X, centroids, scale=1):
    N, D = X.shape
    result = np.zeros((N, len(centroids)))
    for i in range(len(centroids)):
        center = centroids[i]
        col = X - center
        norm = sl.norm(col, ord=2, axis=1)**2
        result[:, i] = np.exp(-norm/(2 * scale**2))

    return result

X_kernelised = kernelise(X, mus)
LG2 = slm.LogisticRegression(C=50).fit(X_kernelised, y)
y_predict2 = LG2.predict(X_kernelised)
print('error rate: ', 1 - np.count_nonzero(y == y_predict2) / len(y)) # should be 0

xgrid_k = kernelise(np.c_[xx_ravel, yy_ravel], mus)
Z2 = LG2.predict(xgrid_k)
Z2 = Z2.reshape(xx.shape)

# plots
fig = plt.figure(figsize=(10.5, 5))
fig.canvas.set_window_title('logregXorDemo')

def plot(index, title, Z):
    Z[Z == 0] = 0.3
    Z[Z == 1] = 0.2 # 针对Pastel2这个colormap进行调色, 与contourf中的vmin和vmax配合使用
    plt.subplot(index)
    plt.title(title)
    plt.axis([-1, 7, -1, 7])
    plt.xticks(np.linspace(0, 6, 7))
    plt.yticks(np.linspace(-1, 7, 9))
    plt.plot(X[y == 0][:, 0], X[y == 0][:, 1], 'b+')
    plt.plot(X[y == 1][:, 0], X[y == 1][:, 1], 'ro', fillstyle='none')
    plt.contourf(xx, yy, Z, cmap='Pastel2', vmin=0, vmax=1)

plot(121, 'poly 10', Z)
plot(122, 'RBF prototypes', Z2)

plt.tight_layout()
plt.show()
