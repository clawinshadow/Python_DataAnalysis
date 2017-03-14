import numpy as np
import scipy.stats as st
import math

print('''
  利用卡方分布来进行两个分类型变量之间的独立性检验，
  自由度为(R - 1)(C - 1), R, C分别为行数和列数
''')

def independence_test(data, alpha):
    
    '''
    列联分析：独立性检验，判断两个分类变量的相关性
    
    Parameters
    ----------
    data : contingency table, 列联表原始数据，二维
    alpha: 执行卡方分布的置信度
    
    Returns
    -------
    tuple(phi, c, v): 为三种相关系数
    '''
    
    shape = np.shape(data)
    if shape[0] < 1 or shape[1] < 1:
        raise ValueError("contingency table size must be 2*2 at least.")

    print("Contingency table: \n", data)
    
    rows     = []      # row No.
    cols     = []
    fo_list  = []      # observe values
    fe_list  = []      # expect values
    fo_fe    = []      # fo - fe
    fo_fe_s  = []      # (fo - fe)**2
    fo_fe_fe = []      # (fo - fe)**2 / fe
    
    row_sum  = np.sum(data, axis=1)      
    col_sum  = np.sum(data, axis=0)
    total_sum = np.sum(data)

    print('Row Sum: ', row_sum)
    print('Col Sum: ', col_sum)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            fo = data[i, j]
            fe = row_sum[i] * col_sum[j] / total_sum
            
            rows.append(i)
            cols.append(j)
            fo_list.append(fo)
            fe_list.append(fe)

    fos = np.array(fo_list)
    fes = np.array(fe_list)
    
    fo_fe = np.round_(fos - fes, decimals=4)
    fo_fe_s = np.round_((fos - fes)**2, decimals=4)
    fo_fe_fe = np.round_((fos - fes)**2/fes, decimals=4)

    print('{0:-^97}'.format(''))
    print('{0:^10}|{1:^10}|{2:^10}|{3:^10}|{4:^15}|{5:^15}|{6:^20}'.format(\
        'Row No.', 'Col No.', 'fo', 'fe', 'fo - fe', '(fo - fe)**2', '(fo - fe)**2 / fe'))
    print('{0:-^97}'.format(''))
    for x in range(len(rows)):
        print('{0:^10}|{1:^10}|{2:^10}|{3:^10}|{4:^15}|{5:^15}|{6:^20}|'.format(\
        rows[x], cols[x], fo_list[x], fe_list[x], fo_fe[x], fo_fe_s[x], fo_fe_fe[x]))

    chi_square = np.sum(fo_fe_fe)
    df = (shape[0] - 1) * (shape[1] - 1)
    rv = st.chi2(df)
    ppf = rv.ppf(1 - alpha)  # 都是单侧检验
    print('Test Value: {0}, Point Value: {1}'.format(chi_square, ppf))

    phi = math.sqrt(chi_square / total_sum)
    c = math.sqrt(chi_square / (chi_square + total_sum))
    v = math.sqrt(chi_square / (total_sum * np.min([shape[0] - 1, shape[1] - 1])))
    print('phi: {0}, c: {1}, v: {2}'.format(phi, c, v))
    
    return phi, c, v


data = np.array([[52, 64, 24], [60, 59, 52], [50, 65, 74]])
independence_test(data, 0.05)
