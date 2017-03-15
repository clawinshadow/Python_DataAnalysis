import numpy as np
import scipy.stats as st
import math
'''
ANOVA: Analysis of Variance, 方差分析，是检验多个总体均值是否相等的统计方法，
       主要是分析分类型自变量对数值型因变量的影响。
Factor： 因素，或是因子，即所要检验的对象
Treatment: 水平，或是处理，即因素的不同表现

SST: Sum of Squares for Total, 全部观测值与总均值的误差平方和
SSA: Sum of Squares for factor A, 组间平方和，各组均值与总均值的误差平方和
SSE: Sum of Squares fo Error, 组内平方和，每个组内的各观测值与其组均值的误差平方和

df_SST: SST的自由度为 n-1, n为全部观测值的个数
df_SSA: SSA的自由度为 k-1, k为因素水平的个数（组数）
df_SSE: SSE的自由度为 n-k

MST: 总体均方，一般不用
MSA: 组间均方，为 SST/df_SSA
MSE: 组内均方，为 SSE/df_SSE

F = MSA/MSE ~ F(k-1, n-k)
'''

def oneway_anova(data, alpha):
    '''
    单因素方差分析：只涉及一个分类型自变量时的方差分析
    
    Parameters
    ----------
    data : [list1, list2, ... , listn] 该自变量的多个水平的观测值表，不一定是标准2维矩阵
    alpha: 执行F统计量假设检验的置信度
    
    Returns
    -------
    R: 自变量与因变量之间的关系强度， R**2 = SSA / SST
    '''
    mas = [] # 记录每个水平的均值
    mt = 0   # 记录所有观测值的均值
    k = len(data) # 因素水平的个数
    n = 0    # 观测值的个数
    counts = [] # 每个水平内观测值的个数
    obs = [] # 所有观测值
    
    for factor in data:
        n += len(factor)
        counts.append(len(factor))
        mas.append(np.mean(factor))
        for ob in factor:
            obs.append(ob)

    # sumtotal = np.sum(np.multiply(mas, counts))
    mt = np.sum(np.multiply(mas, counts)) / n
    sst = np.sum(np.power(np.subtract(obs, mt), 2)).round(4)
    ssa = np.sum(np.multiply(counts, np.power(np.subtract(mas, mt), 2))).round(4)
    sse = sst - ssa
    msa = (ssa / (k - 1)).round(4)
    mse = (sse / (n - k)).round(4)

    rv = st.f(k-1, n-k)
    f = (msa / mse).round(4)   # 待检验的值
    f_crit = rv.ppf(1 - alpha).round(4)   # 在当前置信度下的F临界值
    p = (1 - rv.cdf(f)).round(4)  # P值

    print('{0:-^97}'.format(''))
    print('{0:^10}|{1:^15}|{2:^10}|{3:^15}|{4:^15}|{5:^15}|{6:^15}'.format(\
        'Source', 'SS', 'df', 'MS', 'F', 'P-Value', 'F crit'))
    print('{0:-^97}'.format(''))
    print('{0:^10}|{1:^15}|{2:^10}|{3:^15}|{4:^15}|{5:^15}|{6:^15}'.format(\
        'factors', ssa, k - 1, msa, f, p, f_crit)) # 组间分析
    print('{0:^10}|{1:^15}|{2:^10}|{3:^15}|'.format('obs', sse, n - k, mse)) # 组内分析
    print('{0:^10}|{1:^15}|{2:^10}|'.format('total', sst, n)) # 总计
    # print(f, f_crit, p)
    return ssa / sst
    


data = [[57,66,49,40,34,53,44],[68,39,29,45,56,51],[31,49,21,34,40],[44,51,65,77,58]]
oneway_anova(data, 0.05)
