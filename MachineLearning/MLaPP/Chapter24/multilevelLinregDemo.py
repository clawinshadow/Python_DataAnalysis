import numpy as np
import scipy.io as sio
import scipy.linalg as sl
import sklearn.linear_model as slm
import matplotlib.pyplot as plt

# load data
data = sio.loadmat('mathDataHoff.mat')
print(data.keys())
y = data['y']
schools = y[:, 0]
school_ids = np.unique(schools)
score = y[:, 3]
ses = y[:, 2]

# for each school, fit with OLS individually, and calculate temp datas
Ws = np.zeros((len(school_ids), 2))
sampleSize = np.zeros(len(school_ids))
sigmahat = np.zeros((len(school_ids)))
XXs = np.zeros((len(school_ids), 2, 2))
Xys = np.zeros((len(school_ids), 2))
ys = []
xs = []
for i in range(len(school_ids)):
    school = school_ids[i]
    idx = schools == school
    yi = score[idx].reshape(-1, 1)
    xi = ses[idx]
    xi = xi - np.mean(xi)  # center x
    X = np.c_[np.ones(len(xi)), xi]

    # calculating temp datas
    XX = np.dot(X.T, X)
    Xy = np.dot(X.T, yi)

    # fit with OLS
    wi = np.dot(sl.inv(XX), Xy)
    Ws[i] = wi.ravel()  # first is intercept, second is slope

    sampleSize[i] = len(xi)
    XXs[i] = XX
    Xys[i] = Xy.ravel()
    ys.append(yi.ravel())
    xs.append(X)
    sigmahat[i] = np.var((yi - np.dot(X, wi)), ddof=1) # unbiased variance

w_avg = np.mean(Ws, axis=0)

