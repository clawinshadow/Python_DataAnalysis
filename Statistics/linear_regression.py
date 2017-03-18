import numpy as np
import scipy.stats as st
import math

def pearson(x, y):
    '''
    求两个数值型变量之间的相关系数，度量线性关系强度
    '''
    if len(x) != len(y):
        raise ValueError('The size of x and y must be the same')

    n = len(x)
    x = np.array(x)
    y = np.array(y)
    sumxy = np.sum(x * y)
    sumx = np.sum(x)
    sumy = np.sum(y)
    sumx2 = np.sum(x**2)
    sumy2 = np.sum(y**2)

    divisor = math.sqrt(n * sumx2 - sumx **2) * math.sqrt(n * sumy2 - sumy **2)
    pearson = (n * sumxy - sumx * sumy ) / divisor
    print('pearson: ', pearson)
    return pearson


def linear_regression(x, y):
    '''用最小二乘法进行一元线性回归'''
    if len(x) != len(y):
        raise ValueError('The size of x and y must be the same')
    
    n = len(x)
    x = np.array(x)
    y = np.array(y)
    sumxy = np.sum(x * y)
    sumx = np.sum(x)
    sumy = np.sum(y)
    sumx2 = np.sum(x**2)
    meanx = np.mean(x)
    meany = np.mean(y)

    beta1 = ((n * sumxy - sumx * sumy) / (n * sumx2 - sumx**2)).round(6)
    beta0 = (meany - meanx * beta1).round(6)
    print('estimated regression equation: y = {0} + {1} * x'.format(beta0, beta1))

y = [0.9,1.1,4.8,3.2,7.8,2.7,1.6,12.5,1.0,2.6,0.3,4.0,0.8,3.5,10.2,3.0,\
     0.2,0.4,1.0,6.8,11.6,1.6,1.2,7.2,3.2]
x = [67.3,111.3,173,80.8,199.7,16.2,107.4,185.4,96.1,72.8,64.2,132.2,58.6,\
     174.6,263.5,79.3,14.8,73.5,24.7,139.4,368.2,95.7,109.6,196.2,102.2]
pearson(x, y)
linear_regression(x, y)
