import numpy as np
import scipy.optimize as so

'''
so.linprog()这个方法主要用于求解线性规划问题，算法只有一个'simplex'，估计是单纯形下山法
Minimize: c^T * x

Subject to: A_ub * x <= b_ub
            A_eq * x == b_eq

从名字就可以看出来，目标函数包括所有的约束条件都是线性的，所以各参数都只包含各种系数矩阵，
不必代入未知参数，下面简单说下各参数的含义：
	
c : 一维数组，目标函数中的系数矩阵
A_ub : 二维数组，行数代表不等式的个数，列数代表未知参数的个数，如果没有某个未知参数，一律用0补齐
b_ub : 一维数组，不必reshape(-1, 1)， 对应每个不等式约束中右边的值
A_eq : 二维数组，与A_ub类似，只不过是等式约束中的系数矩阵
b_eq : 一维数组，与b_ub类似，等式约束中右边的值
bounds : 每个未知参数的边界，没有的话写(None, None)

Returns: 依然是OptimizeResult
'''

# 官网中的例子，挺有代表性的，复杂的可以参考..\MachineLearning\MLaPP\LinearRegression\linregRobustDemoCombined.py
'''
Minimize: f = -1*x[0] + 4*x[1]

Subject to: -3*x[0] + 1*x[1] <= 6
            1*x[0] + 2*x[1] <= 4
            x[1] >= -3
'''

c = [-1, 4]
A = [[-3, 1], [1, 2]]
b = [6, 4]
x0_bounds = (None, None)
x1_bounds = (-3, None)
res = so.linprog(c, A_ub=A, b_ub=b, bounds=(x0_bounds, x1_bounds), options={"disp": True})
print('Optimize Result: ', res)
