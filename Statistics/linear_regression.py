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

x = [0.9,1.1,4.8,3.2,7.8,2.7,1.6,12.5,1.0,2.6,0.3,4.0,0.8,3.5,10.2,3.0,\
     0.2,0.4,1.0,6.8,11.6,1.6,1.2,7.2,3.2]
y = [67.3,111.3,173,80.8,199.7,16.2,107.4,185.4,96.1,72.8,64.2,132.2,58.6,\
     174.6,263.5,79.3,14.8,73.5,24.7,139.4,368.2,95.7,109.6,196.2,102.2]
pearson(x,y)
