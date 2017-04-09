import math
import numpy as np
import scipy.linalg as sl
import scipy.spatial.distance as sd

'''
demos about various distances used in machine-learning algorithms
1. Minkowski Distance
   d = power(sum(x1 - x2)**p), 1/p)
   当p = 1时，就是曼哈顿距离
   当p = 2时，就是欧氏距离
   当p = inf时，就是切比雪夫距离

2. Euclidean Distance
   欧式距离是最常用的一种距离算法，源自欧式空间中两点间的距离公式
   d = math.sqrt(sum(x1 - x2)**2)

3. 曼哈顿距离 Manhattan Distance
   即在城市中开车行驶，没法走直线的距离，即城市街区距离 city block distance
   d = sum(abs(x1 - x2))

4. Chebshev Distance 切比雪夫距离
   典型的实例是国际象棋的移动距离
   d = max(abs(x1 - x2))

5. 夹角余弦 Cosine
               dot(x, y)
   Cosθ = -------------------
            norm(x) * norm(y)

6. 汉明距离 Hamming Distance
   两个等长字符串s1与s2之间的汉明距离定义为将其中一个变为另一个所需要的最小替换次数

7. 杰卡德相似系数 Jaccard Similarity Coefficient
   两个集合A与B的交集元素在A和B的并集中所占的比例
              |A & B|
   J(A, B) = ---------
              |A U B|

   那么杰卡德距离就是用1 减去 杰卡德相似系数
   d = 1 - J(A, B)

8. 马氏距离 Mahalanobis Distance 印度人
   有M个样本向量X1~Xm, 协方差矩阵为S，均值记为向量μ， 则样本向量Xi到μ的马氏距离为：
   D(Xi) = math.sqrt((Xi - μ)*S.inv*(Xi - μ))
   其中两个向量之间的马氏距离为:
   D(Xi, Xj) = math.sqrt((Xi - Xj)*S.inv*(Xi - Xj))
   那么若协方差矩阵为单位矩阵，即各向量之间独立同分布，则马氏距离变成了欧氏距离
   D(Xi, Xj) = math.sqrt((Xi - Xj)*(Xi - Xj))
'''

def Minkowski_d(u, v, p):
    x = np.absolute(np.subtract(u, v))
    return np.power(np.sum(x**p), 1/p)

def Chebshev_d(u, v):
    x = np.absolute(np.subtract(u, v))
    return np.max(x)

def Cosine(u, v):
    return np.dot(u, v) / (sl.norm(u) * sl.norm(v))

def Hamming_d(u, v):
    x = np.absolute(np.subtract(u, v))
    d = np.nonzero(x)
    return np.shape(d)[1]

def M_d(u, v, VI):
    x = np.subtract(u, v)
    return math.sqrt(np.dot(np.dot(x.T, VI), x))

u = [1, 2, 3]
v = [4, 5, 6]
print('u: ', u)
print('v: ', v)
print('Mahattan Distance: ', Minkowski_d(u, v, 1))
print('Mahattan Distance by Scipy: ', sd.minkowski(u, v, 1))
print('City Block Distance by Scipy: ', sd.cityblock(u, v))
print('Euclidean Distance: ', Minkowski_d(u, v, 2))
print('Euclidean Distance by Scipy: ', sd.euclidean(u, v))
print('Minkowski Distance 3: ', Minkowski_d(u, v, 3))
print('Minkowski Distance 3 by Scipy: ', sd.minkowski(u, v, 3))
v = [4, 7, 5]
print('{0:-^60}'.format('Seperate Line'))
print('v: ', v)
print('Chebshev Distance: ', Chebshev_d(u, v))
print('Chebshev Distance by Scipy: ', sd.chebyshev(u, v))
print('Cosine: ', Cosine(u, v))
print('Cosine by Scipy: ', sd.cosine(u, v))  # while in scipy, it's 1-Cosine(u, v) actually
u = [1, 1, 0, 1, 0, 1, 0, 0, 1]
v = [0, 1, 1, 0, 0, 0, 1, 1, 1]
print('{0:-^60}'.format('Seperate Line'))
print('u: ', u)
print('v: ', v)
print('Hamming Distance: ', Hamming_d(u, v))
print('Hamming Distance by Scipy: ', sd.hamming(u, v)) # while in scipy. it's Hamming_d / n, 6/9 in this example
print('Jaccard Distance in Scipy: ', sd.jaccard(u, v))

# scipy里面的关于协方差矩阵的参数习惯写法很奇怪
# 每个一维数组其实是一个特征向量的观测值集合，并不是一个样本的每个特征值向量集合
# 所以样本应该是u[i], v[i]，使用起来一定要注意
u = [88.5, 96.8, 104.1, 111.3, 117.7, 124.0, 130, 135.4, 140.2, 145.3, 151.9, 159.5, 165.9, 169.8,\
     171.6, 172.3, 172.7]
v = [12.54, 14.65, 16.64, 18.98, 21.26, 24.06, 27.33, 30.46, 33.74, 37.69, 42.49, 48.08, 53.37, \
     57.08, 59.35, 60.68, 61.40]
featureMat = np.array([u, v])
cov = np.cov(featureMat)
covinv = sl.inv(cov)
print('{0:-^60}'.format('Seperate Line'))
print('u: ', u)
print('v: ', v)
print('Mahalanobis Distance: ', M_d([u[0], v[0]], [u[1], v[1]], covinv))
print('Mahalanobis Distance by Scipy: ', sd.mahalanobis([u[0], v[0]], [u[1], v[1]], covinv))
    
