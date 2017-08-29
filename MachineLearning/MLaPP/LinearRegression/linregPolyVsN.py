import math
import numpy as np
import sklearn.linear_model as slm
import sklearn.preprocessing as sp
import matplotlib.pyplot as plt

def DM(x, degree):
    if degree < 1:
        return x
    else:
        result = x.reshape(-1, 1)
        i = 2
        while i <= degree:
            result = np.c_[result, (x**i).reshape(-1, 1)]
            i += 1
        return result

def Noisy(x):
    sigma = 2         # 噪声的标准差 
    w_true = [-1.5, 1/9]
    y_true = w_true[0] * x + w_true[1] * x**2       # real func，没有带截距

    return y_true + sigma * np.random.randn(len(x))

def GenerateTrainingData(n):
    x_train = np.linspace(0, 20, n)
    y_train = Noisy(x_train)

    return x_train, y_train

def GetTrainAndTestMSE(n, degree):
    x_train, y_train = GenerateTrainingData(n)
    x_train_dm = DM(x_train, degree)
    x_train_dm = sp.StandardScaler().fit_transform(x_train_dm)  # center the data
    
    x_test = np.arange(0, 20, 0.1)  # 测试集保持不变
    y_test = Noisy(x_test)          # 作为测试集里面y的值也要加上噪声，如此才符合真实情况
    x_test_dm = DM(x_test, degree)
    x_test_dm = sp.StandardScaler().fit_transform(x_test_dm)
    
    model = slm.Ridge(alpha=0, fit_intercept=False)          # 岭回归相比OLS有更好的数值稳定性
    model.fit(x_train_dm, y_train)

    y_train_predict = model.predict(x_train_dm)
    y_test_predict = model.predict(x_test_dm)

    trainMSE = np.mean((y_train_predict - y_train)**2)
    testMSE = np.mean((y_test_predict - y_test)**2)

    return trainMSE, testMSE    

degrees = [1, 2, 10, 25]        # 四个子图形分别使用到的degree

def Draw(index, degree):
    ns = np.arange(8, 201, 12)
    trainMSEs = []
    testMSEs = []
    for i in range(len(ns)):
        n = ns[i]
        trainMSE, testMSE = GetTrainAndTestMSE(n, degree)
        trainMSEs.append(trainMSE)
        testMSEs.append(testMSE)

    print('test MSEs with degree {0}: {1}'.format(degree, testMSEs))
    plt.subplot(index)
    plt.axis([0, 200, 0, 22])
    plt.xticks(np.arange(0, 201, 20))
    plt.yticks(np.arange(0, 22.1, 2))
    plt.xlabel('size of training set')
    plt.ylabel('mse')
    plt.title('truth=degree 2, model = degree {0}'.format(degree), fontdict={ 'fontsize': 10 })
    plt.hlines(4, 0, 200, lw=2)  # 之所以是4，因为noise floor是sigma**2=4
    plt.plot(ns, trainMSEs, ls=':', marker='s', mew=2, fillstyle='none', color='midnightblue', label='train')
    plt.plot(ns, testMSEs, ls='-', marker='x', mew=2, fillstyle='none', color='red', label='test')
    plt.legend()

fig = plt.figure(figsize=(12, 11))
fig.canvas.set_window_title('linregPolyVsN')

Draw(221, degrees[0])
Draw(222, degrees[1])
Draw(223, degrees[2])
Draw(224, degrees[3])

plt.show()

    
    
