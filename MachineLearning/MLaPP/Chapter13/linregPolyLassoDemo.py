import numpy as np
import scipy.linalg as sl
import sklearn.preprocessing as sp
import matplotlib.pyplot as plt

#rs = np.random.RandomState(654321)
np.random.seed(654321)

# prepare data
deg = 14
w = np.array([-1.5, 1/9])  # true w
sigma = 2 # noise std
x_train = np.linspace(0, 20, 21)
x_test = np.arange(0, 20, 0.1)
y_train = w[0] * x_train + w[1] * x_train**2 + sigma * np.random.randn(1, len(x_train))
y_test_noisyfree = w[0] * x_test + w[1] * x_test**2
y_test = y_test_noisyfree + sigma * np.random.randn(1, len(x_test))

# remove the mean of response
y_train = (y_train - np.mean(y_train)).reshape(-1, 1)
y_test = (y_test - np.mean(y_test)).reshape(-1, 1)
# preprocessing X
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)
ss = sp.StandardScaler().fit(x_train)                    # 1. standardize first
x_train = ss.transform(x_train)
mm = sp.MinMaxScaler(feature_range=(-1, 1)).fit(x_train) # 2. rescale to [-1, 1], 参数却是(0, 1)
x_train = mm.transform(x_train)
po = sp.PolynomialFeatures(degree=deg).fit(x_train)      # 3. polynomial expansion
x_train = po.transform(x_train)[:, 1:]                   # remove the first column [1, 1, 1, 1...]

print(x_train.shape)
x_test = ss.transform(x_test)
x_test = mm.transform(x_test)
x_test = po.transform(x_test)[:, 1:]
print(x_test.shape)
print(np.mean(x_train, axis=0))
print(np.mean(y_train))

lambda_max = sl.norm(2 * np.dot(x_train.T, y_train), np.inf)
print(lambda_max)
