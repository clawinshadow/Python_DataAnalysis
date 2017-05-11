Normalization: 将一组数据按最大值和最小值进行等比例缩放到某个区间，比如[-1, 1]，相当于sklearn中的range_scaling

Standardization: 将一组数据按高斯分布标准化，不管它的实际分布是什么样的，都转化成高斯分布的典型特征，即均值为0，标准差为1.

                 所以这个标准化过程也称之为mean removal & variance scaling. 即均值移除与方差缩放，
                 
                 假设数组A：[a1, a2, a3, ..., an], A的均值为Mean(A), 方差为Var(A), 具体算法如下：
                 
                            a[i] - Mean(A)
                  a[i] = ---------------------
                           math.sqrt(Var(A))
                           
Regularization: 正则化，它是避免过拟合的一种技术，以PRML中的第一节曲线拟合为例，给出一组数据点，我们可以用高阶的多项式来完美拟合训练数据集。但这种拟合过多的迎合了训练数据集中的各种噪声而忽略了数据的真实规律，所以我们为了避免这种情况，加入了一个惩罚函数:
                
                lambda * ||w||**2
                
                来平衡多项式系数不至于过大，所以这就是regularization, 如果惩罚函数使用了参数的一阶范数那就是 L1 Regularization, 如果是二阶范数那就是L2 Regualarization
