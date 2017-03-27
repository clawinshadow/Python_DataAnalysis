import numpy as np
import scipy.linalg as sl

'''
计算矩阵的伪逆矩阵 Moore-Penrose pseudoinverse
用于当矩阵A的秩 r < n 时，构造A的一个广义逆矩阵

当A为非奇异n * n 矩阵，且奇异值分解为 A = U*Σ*V.T 时
A的逆矩阵A.inv = V*Σ.inv*U.T
更一般地，若A为秩为r的m*n矩阵，则矩阵Σ将为如下形式的m*n矩阵
     Σ1 | O 
Σ =  -------
      O | O

     σ1
Σ1 =   σ2
         σ3
           ...
             σr

显然 Σ 是一个奇异矩阵，不具备标准意义上的逆矩阵
但我们可定义

     1/σ1        |
       1/σ2      |
         1/σ3    | O
Σ+ =       ...   |
             1/σr|
     -----------------
           O     | O
然后 A+ = V*Σ+*U.T 为A的伪逆矩阵，它满足以下四个条件(彭罗斯条件)：
1. AXA = A
2. XAX = X
3. (AX).T = AX
4. (XA).T = XA
并且A+是唯一的满足这些条件的n*m矩阵

伪逆矩阵可以非常方便的用来求解最小二乘问题
1. 当A为m*n矩阵，且rank(A) = n 时
    Σ1              Σ1+
Σ = ---  那么 Σ+ =  ---
     O               O

最小二乘解 x = A+*b = V*Σ+*U.T*b
2. 当rank(A) = r < n 时，此时最小二乘解有无穷多个，但我们可以求解最小范数解
此时也是用伪逆来求解 x = A+ * b

'''

A = np.array([[1, 5], [1, 3], [1, 11], [1, 5]])
b = np.array([1, -1, 3, 5], np.newaxis)
print('A : \n', A)
print('b : \n', b)
print('A.pseudoinverse: \n', sl.pinv(A))
# print('A.pseudoinverse2: \n', sl.pinv2(A))
print('lsd solution of Ax = b: x = ', np.dot(sl.pinv(A), b))
print('x == [0, 1/3]: ', np.allclose(np.dot(sl.pinv(A), b), np.array([0, 1/3])))
