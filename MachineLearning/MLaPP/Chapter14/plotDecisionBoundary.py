import numpy as np
import sklearn.metrics.pairwise as smp
import sklearn.preprocessing as spp
import matplotlib.pyplot as plt

'''
slightly different with in books, coz the algorithm details is a bit different between sklearn & pmtk

Used for plot decision boundary of classification problems, briefly 2 steps:
1. plot the boundary using plt.contour(), just one line in binary classification problem
2. fill color in the 2 regions, by plot massive points with different colors, looks like a filled color region finally

the difference of nerr is because the model.predict(), > 0.5 or >= 0.5,
as long as the coef_ is the same as in matlab codes, predict() diffenrece does not matter
'''

def DataWindow(X):
    '''get axis range and intervals base on min and max of X'''
    N, D = X.shape
    assert D == 2
    minX1 = np.min(X[:, 0])
    maxX1 = np.max(X[:, 0])
    minX2 = np.min(X[:, 1])
    maxX2 = np.max(X[:, 1])
    dx1 = 0.15 * (maxX1 - minX1)  # tolerant range of X axis
    dx2 = 0.15 * (maxX2 - minX2)  # tolerant range of Y axis
    window = np.array([minX1 - dx1, maxX1 + dx1, minX2 - dx2, maxX2 + dx2])

    return window

def GridPredict(dataRange, resolution, model, preprocessFunc, **kwargs):
    '''model must have a predict function: model.predict(X)'''
    xs = np.linspace(dataRange[0], dataRange[1], (int)(resolution))
    ys = np.linspace(dataRange[2], dataRange[3], (int)(resolution))
    xx, yy = np.meshgrid(xs, ys)
    xx_ravel = xx.ravel()
    yy_ravel = yy.ravel()
    zs = np.c_[xx_ravel, yy_ravel]  # N * 2

    # preprocess data, kernelise / poly
    # 1. standardize
    if 'sscaler' in kwargs:
        sscaler = kwargs['sscaler']
        zs = sscaler.transform(zs)
        zs = zs * kwargs['factor']
    # 2. rescale
    if 'scaler' in kwargs:
        scaler = kwargs['scaler']
        zs = scaler.transform(zs)
    # 3. kernel or poly
    if preprocessFunc != 'none':
        if preprocessFunc.__name__ == 'rbf_kernel':
            rbf_scale = kwargs['rbfscale']
            centers = kwargs['centers']
            gamma = 1 / (2 * rbf_scale ** 2)
            zs = smp.rbf_kernel(zs, centers, gamma=gamma)  # kernelise
            zs = zs / np.sqrt(2 * np.pi * rbf_scale ** 2)
        if preprocessFunc.__name__ == 'poly':
            deg = kwargs['deg']
            zs = preprocessFunc(zs, deg)

    zz = model.predict(zs)
    zz = zz.reshape(xx.shape)
    print(np.count_nonzero(zz == 1))

    return xx, yy, zz

def plot(X, y, model, preprocessFunc, **kwargs):
    '''only support binary classification problem'''
    classes = np.unique(y)
    # colors = [np.array([55, 155, 255]) / 255, np.array([255, 128, 0]) / 255] # lightblue & orange
    colors = ['darkblue', 'darkorange']
    markers = ['+', 'o']
    marker_colors = ['blue', 'orange']
    dataRange = DataWindow(X)
    resolution = 300
    xx_grid, yy_grid, zz_grid = GridPredict(dataRange, resolution, model, preprocessFunc, **kwargs)  # for plot contour line
    xx_sparse, yy_sparse, zz_sparse = GridPredict(dataRange, resolution/2.5, model, preprocessFunc, **kwargs)  # for plot points

    if 'subplotIndex' in kwargs:
        plt.subplot(kwargs['subplotIndex'])
    else:
        plt.subplot()

    if 'markers' in kwargs:
        markers = kwargs['markers']
    if 'mcolors' in kwargs:
        marker_colors = kwargs['mcolors']
    if 'title' in kwargs:
        plt.title(kwargs['title'])
    if 'tickRange' in kwargs:
        tRange = kwargs['tickRange']
        plt.xticks(np.linspace(tRange[0], tRange[1], tRange[1] - tRange[0] + 1))
        plt.yticks(np.linspace(tRange[2], tRange[3], tRange[3] - tRange[2] + 1))

    plt.plot(xx_sparse[zz_sparse == classes[0]], yy_sparse[zz_sparse == classes[0]],\
             marker='.', ms=0.1, color=colors[0], linestyle='none')
    plt.plot(xx_sparse[zz_sparse == classes[1]], yy_sparse[zz_sparse == classes[1]],\
             marker='.', ms=0.1, color=colors[1], linestyle='none')
    plt.contour(xx_grid, yy_grid, zz_grid, colors='k', linewidths=2)
    plt.plot(X[y == classes[0]][:, 0], X[y == classes[0]][:, 1], marker=markers[0],\
             color=marker_colors[0], mew=2, fillstyle='none', linestyle='none')
    plt.plot(X[y == classes[1]][:, 0], X[y == classes[1]][:, 1], marker=markers[1],\
             color=marker_colors[1], mew=2, fillstyle='none', linestyle='none')

    if 'drawSV' in kwargs:
        if 'svIndices' in kwargs:
            SV_indices = kwargs['svIndices']
        else:
            SV_indices = np.abs(model.coef_) > 1e-5
        SV = X[SV_indices.ravel()]  # support vectors
        plt.plot(SV[:, 0], SV[:, 1], 'ko', ms=10, mew=1, linestyle='none', fillstyle='none')

