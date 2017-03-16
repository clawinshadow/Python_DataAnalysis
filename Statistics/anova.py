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
SSE: Sum of Squares for Error, 组内平方和，每个组内的各观测值与其组均值的误差平方和
SSR: Sum of Squares for Rows, 行因素产生的误差平方和
SSC: Sum of Squares for Columns, 列因素产生的误差平方和

df_SST: SST的自由度为 n-1, n为全部观测值的个数
df_SSA: SSA的自由度为 k-1, k为因素水平的个数（组数）
df_SSE: 单因素方差分析中SSE的自由度为 n-k, 双因素中为(k-1)(r-1)
df_SSR: SSR的自由度为 k-1 
df_SSC: SSE的自由度为 r-1

MST: 总体均方，一般不用
MSA: 组间均方，为 SST/df_SSA
MSE: 组内均方，为 SSE/df_SSE
MSR: 行间均方，为 SSR/df_SSR
MSC: 列间均方，为 SSC/df_SSC

F = MSA/MSE ~ F(k-1, n-k)
Fr = MSR/MSE ~ F(k-1, (k-1)(r-1))
Fc = MSC/MSE ~ F(r-1, (k-1)(r-1))

LSD: 最小显著性差异检验， Least significant difference.
     用于检测到底是哪两个水平之间有显著差异
     
     LSD = t((1+alpha)/2)*math.sqrt(MSE*(1/ni+1/nj))
     ni, nj 分别为待检验两个水平的观测值个数， 共要进行Combine(k, 2)次检验
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

    rv = st.t(n - k)  # for LSD
    t = rv.ppf(1 - alpha / 2)
    for i in range(len(mas)):
        for j in range(len(mas)):
            if j > i:
                ni, nj = counts[i], counts[j]
                lsd = t * math.sqrt(mse * (1 / ni + 1 / nj))
                diff = abs(mas[i] - mas[j]) # diff > lsd 意味着这两个水平存在显著差异
                print('({0}, {1}): LSD: {2}, Diff: {3}'.format(i + 1, j + 1, lsd, diff))
                    
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
    

def twoway_anova(data, alpha):
    '''
    双因素方差分析：涉及两个分类型自变量时的方差分析，这里只写一份无重复双因素的代码
    
    Parameters
    ----------
    data : R(m*n) 该自变量的多个水平的观测值表，标准的 m * n 二维矩阵
    alpha: 执行F统计量假设检验的置信度
    
    Returns
    -------
    R: 自变量与因变量之间的关系强度， R**2 = SSR + SSC / SST
    '''
    data = np.array(data)
    k, r = np.shape(data)           # k为行数，r为列数
    n = k * r                       # 总观测值
    mrs = np.mean(data, axis=1)     # 每行的均值
    mcs = np.mean(data, axis=0)     # 每列的均值
    mt = np.mean(data)              # 总体均值
    
    sst = (np.var(data) * k * r).round(4)       # 总体平方和
    ssr = (np.sum((mrs - mt)**2) * r).round(4)  # 行间平方和
    ssc = (np.sum((mcs - mt)**2) * k).round(4)  # 列间平方和
    sse = (sst - ssr - ssc).round(4)            # 随机误差
    msr = (ssr / (k - 1)).round(4)
    msc = (ssc / (r - 1)).round(4)
    mse = (sse / ((k - 1) * (r - 1))).round(4)

    rvr = st.f(k - 1, (k - 1) * (r - 1))
    fr = (msr / mse).round(4)                 # 检验行因素
    fr_crit = rvr.ppf(1 - alpha).round(4)     # 在当前置信度下的F临界值
    pr = (1 - rvr.cdf(fr)).round(4)           # P值

    rvc = st.f(r - 1, (k - 1) * (r - 1))
    fc = (msc / mse).round(4)                 # 检验行因素
    fc_crit = rvc.ppf(1 - alpha).round(4)     # 在当前置信度下的F临界值
    pc = (1 - rvc.cdf(fc)).round(4)           # P值
                    
    print('{0:-^97}'.format(''))
    print('{0:^10}|{1:^15}|{2:^10}|{3:^15}|{4:^15}|{5:^15}|{6:^15}'.format(\
        'Source', 'SS', 'df', 'MS', 'F', 'P-Value', 'F crit'))
    print('{0:-^97}'.format(''))
    print('{0:^10}|{1:^15}|{2:^10}|{3:^15}|{4:^15}|{5:^15}|{6:^15}'.format(\
        'Rows', ssr, k - 1, msr, fr, pr, fr_crit))
    print('{0:^10}|{1:^15}|{2:^10}|{3:^15}|{4:^15}|{5:^15}|{6:^15}'.format(\
        'Cols', ssc, r - 1, msc, fc, pc, fc_crit))  
    print('{0:^10}|{1:^15}|{2:^10}|{3:^15}|'.format('Errors', sse, (k - 1) * (r - 1), mse)) 
    print('{0:^10}|{1:^15}|{2:^10}|'.format('Total', sst, k * r - 1)) 

    return (ssr + ssc) / sst


data = [[57,66,49,40,34,53,44],[68,39,29,45,56,51],[31,49,21,34,40],[44,51,65,77,58]]
oneway_anova(data, 0.05)

data = [[365,350,343,340,323],
        [345,368,363,330,333],
        [358,323,353,343,308],
        [288,280,298,260,298]]
twoway_anova(data, 0.05)
