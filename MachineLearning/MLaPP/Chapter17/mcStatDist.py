import numpy as np
import scipy.linalg as sl
import scipy.sparse as spa

'''
Refer to Page.597
how to calculate a stationary distribution of Markov Chain, in a more general approach

1. input is the transition matrix, A
2. temp = I - A
3. M = replace the last column of temp with 1 (actually arbitrary column is ok)
4. M is absolutely not a singular matrix, then define a vector r = [0, 0, ...., 1]
   the index of the only 1 in r should be related to the replaced column in M
5. solve π*M = r, that is π = np.dot(r, sl.inv(M))

if we directly solve the eigen problem A.T * v = v, we will face numerical issue if any element
in A is 0 or 1, that is we require the A(i, i) must greater than 0, but it obviously is an 
over-strict constraint, so it's not feasible generally. 
'''

def stationary(A):
    N, D = A.shape   # N * N
    assert N == D
    M = np.eye(N) - A
    M[:, -1] = 1     # replace the last column with all 1
    r = np.zeros(N)
    r[-1] = 1     # replace the last zero with 1

    return np.dot(r, sl.inv(M))

def powermethod(G):
    '''for PageRank
    1. 我们的input是一个Graph, 表示有向图结构的矩阵，G(i, j) = 1 表示从节点 j 到 i 有一条link
    2. 因为这个G矩阵不太可能是一个能构成稳态分布的transition matrix,所以我们要做一些假设，将其转化为
       可以求稳态分布的transition matrix, 假设我们在这个有向图上random walk, 在每个时间节点上
       - 以一个较大的概率 p, 假设 p = 0.85, 我们会选择沿着这个节点出去的任意一个link走下去
       - 同样的，以较小的概率 1- p = 0.15, 我们会无视link，选择跳到任意一个节点
       有了这两个假设后，这个随机过程就变成irreducible, 有了稳态分布的充分必要条件
       用数学公式抽象出来如下：

                  p * G(i, j) / c(j) + δ     if: c(j) != 0
       M(i, j) =
                  1/ n                       if: c(j) == 0

       其中：δ = (1 - p) / n 就是无视outlink，跳到任意一个节点的概率
            c(j) 是一个节点的出度 out-degree, 即正常沿着出去的节点走，总共有几条路可以选

       经过这样的转化后，就将一个表示网页的有向图结构，转化成了一个具有稳态分布的transition matrix

    3. 为了方便计算，以及节省存储空间，我们将M的表示更一步简化

       D: is a diagonal matrix, with entries:

                    1 / c(j)   if c(j) != 0
          d(j, j) =
                    0          if c(j) == 0


       Z: is a vector

                  δ            if c(j) != 0
          z(j) =
                  1/ n         if c(j) == 0

       M = p * G * D + 1 * Z.T

    4. 原则上我们就简单的求解 v = M*v， v = π.T, 就可以得到稳态分布，
       但我们就用迭代的方法，乘以M多次自然就收敛了
    '''
    N, D = G.shape
    assert N == D
    p = 0.85
    delta = (1 - p) / N

    C = np.sum(G, axis=0)
    condition = [C == 0, C != 0]
    choices = [0, 1 / C]
    choices2 = [1 / N, delta]
    D = np.select(condition, choices)
    D = np.diag(D.ravel())
    Z = np.select(condition, choices2)
    I = np.ones(N).reshape(-1, 1)

    # use sparse matrix instead, or it will probably exhaust memory
    G_csr = spa.csr_matrix(G)
    D_csr = spa.csr_matrix(D)
    I_csr = spa.csr_matrix(I)
    Z_csr = spa.csr_matrix(Z.reshape(1, -1))

    # M = p * np.dot(G, D) + np.dot(I, Z.reshape(1, -1))
    M = p * np.dot(G_csr, D_csr) + np.dot(I_csr, Z_csr)
    M = M.toarray()
    # print(M)

    maxIter = 50
    pi = np.tile(1/N, N).reshape(-1, 1)
    for i in range(maxIter):
        pi_new = np.dot(M, pi)
        pi_new = pi_new / np.sum(pi_new)
        if np.allclose(pi, pi_new):
            print('Converged!')
            break

        # print(pi_new.ravel())
        pi = pi_new

    return pi.ravel()

def Demo():
    A = np.array([[0, 1, 0],
                  [0.5, 0, 0.5],
                  [1, 0, 0]])
    print(stationary(A))   # Figure 17.4(a)

    A = np.array([[0, 0.5, 0, 0, 0, 0.5],
                  [0, 0, 0.5, 0.5, 0, 0],
                  [0, 0, 0, 1/3, 1/3, 1/3],
                  [1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0]], dtype='float64')  # Figure 17.5
    b = stationary(A)
    # 与书里面的结果不一致。。why? 因为这个A不是一个可以达到稳态分布的transition matrix
    # 要计算这种类似网页的PageRank需要用到另外的算法，不能简单的作为一个求稳态分布的问题去求解
    print(b)

    G = np.array([[0, 0, 0, 1, 0, 1],
                  [1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 1, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [1, 0, 1, 0, 0, 0]], dtype='float64')  # Figure 17.5    G
    pi = powermethod(G)  # 这个才是对的，与书里面的一致
    print(pi)

# Demo()