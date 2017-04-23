import numpy as np
import sklearn.metrics as sm

'''
前面的都是对分类问题模型的评估，这一节是对回归模型的评估分析
主要就是那几个指标，总平方和SST，回归平方和SSR, 以及残差平方和SSE
SST = SSR + SSE
总平方和SST：即样本的离差平方和，每个观测值减去观测均值的平方和
回归平方和SSR：每个回归值减去观测均值的平方和，这部分平方和是可由自变量和因变量之间的线性关系所解释的
残差平方和SSE：每个回归值减去观测值的平方和，是x和y的线性影响之外的因素引起的误差，不可解释的随机误差。

R2：判定系数，R2 = SSR/SST, 因为SSR是由线性关系引起的，所以这个比例越大，表示误差中线性关系的占比越大，
    相关性越明显
'''

y_true = [3, -0.5, 2, 7]    # 实际值
y_pred = [2.5, 0.0, 2, 8]   # 观测值

meanY = np.mean(y_true)
SST = np.sum(np.power(np.subtract(y_true, meanY), 2))
SSR = np.sum(np.power(np.subtract(y_pred, meanY), 2))
SSE = np.sum(np.power(np.subtract(y_pred, y_true), 2))
print('y_true: ', y_true)
print('y_pred: ', y_pred)
# 对非最小线性二乘的回归来说，SST = SSR + SSE 并不一定成立
print('SST, SSR, SSE: ', SST, SSR, SSE) 

# 但无论SST是否等于SSR+SSE， Explained variance score 一定是等于 1 - Var(y_true)/Var(y_true - y_pred)
VarY = np.var(y_true)  # 求和后除以4，不是除以3
VarY_Yi = np.var(np.subtract(y_true, y_pred))
evs = 1 - VarY_Yi / VarY
print(evs)
print('VarY, Var(Y - Yi): ', VarY, VarY_Yi)
evs_sklearn = sm.explained_variance_score(y_true, y_pred)  
print('evs, evs by sklearn: ', evs, evs_sklearn)
print('evs == evs_sklearn: ', np.allclose(evs, evs_sklearn))

# Mean Absolute Error, MAE, 观测值与实际值之差的绝对值之和 再除以样本数量
mae = sm.mean_absolute_error(y_true, y_pred)
print('Mean Absolute Error: ', mae)

# Mean Squared Error， 观测值与实际值之差的平方和 再除以样本数量
mse = sm.mean_squared_error(y_true, y_pred)
print('Mean Squared Error: ', mse)

# R2 score, 1 - SSE / SST
r2_me = 1 - SSE / SST
r2 = sm.r2_score(y_true, y_pred)
print('r2_me, r2 by sklearn: ', r2_me, r2)
print('r2_me == r2: ', np.allclose(r2, r2_me))
