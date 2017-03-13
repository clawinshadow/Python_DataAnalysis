import numpy as np
import scipy.stats as st
import math

print('''
  利用卡方分布来进行两个分类型变量之间的独立性检验，
  自由度为(R - 1)(C - 1), R, C分别为行数和列数
''')

def independence_test(data):
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
    row_sum  = []      
    col_sum  = []
    total_sum = np.sum(data)
    for i in range(shape[0]):
        row_sum.append(np.sum(data[i]))
    for j in range(shape[1]):
        col_sum.append(np.sum(data[:,j]))

    print(row_sum)
    print(col_sum)
    for i in range(shape[0]):
        for j in range(shape[1]):
            fo = data[i, j]
            fe = row_sum[i] * col_sum[j] / total_sum
            
            rows.append(i)
            cols.append(j)
            fo_list.append(fo)
            fe_list.append(fe)
            fo_fe.append(fo - fe)
            fo_fe_s.append((fo - fe)**2)
            fo_fe_fe.append((fo - fe)**2 / fe)

    print('{0:^10}{1:^10}{2:^10}{3:^10}{4:^15}{5:^15}{6:^20}'.format(\
        'Row No.', 'Col No.', 'fo', 'fe', 'fo - fe', '(fo - fe)**2', '(fo - fe)**2 / fe'))
    for x in range(len(rows) - 1):
        print('{0:^10}{1:^10}{2:^10}{3:^10}{4:^15}{5:^15}{6:^20}'.format(\
        rows[x], cols[x], fo_list[x], fe_list[x], fo_fe[x], fo_fe_s[x], fo_fe_fe[x]))

    testvalue = np.sum(fo_fe_fe)
    df = (shape[0] - 1) * (shape[1] - 1)
    rv = st.chi2(df)


data = np.array([[52, 64, 24], [60, 59, 52], [50, 65, 74]])
independence_test(data)
