import numpy as np
import sklearn.discriminant_analysis as sda

'''
参考资料：1. 《多元统计分析》 第四章 判别分析
          2. 维基百科：https://en.wikipedia.org/wiki/Linear_discriminant_analysis
          3. sklearn doc: http://scikit-learn.org/stable/modules/lda_qda.html

判别分析的核心是马氏距离，假设待判别向量为 x, 样本中共有k个分类，每个分类的数据集为 Gi, i = 1, 2, ..., k;
每个分类的样本均值为μ1, μ2, ..., μk 协方差矩阵为Σ1, Σ2, ..., Σk

则 x 到各个样本均值的马氏距离为 M_d = (x - μi).T * Σi.inv * (x - μi), 判别分析的核心思想就是计算x到每个分类的
样本均值的马氏距离，选取最近的一个作为它的分类

1. 假如各分类的协方差矩阵均不相同，则每两个分类的马氏距离之差:

   W(x) = (x - μi).T * Σi.inv * (x - μi) - (x - μj).T * Σj.inv * (x - μj)
   
   i, j 是 k 中任意两个分类，那么所有情况有 k(k-1)/2 种，从中选取一个i，当所有j不等于i的时候，W(x)均小于零，
   意味着 x 到 μi 的距离是最近的，所以与别的均值的距离之差总是小于零，这个i分类就是判别分析的分类结果。

   因为判别函数W(x)是一个二次型函数，所以此时的判别分析也称为二次判别分析，Quadratic Disciminant Analysis(QDA)

2. 假如各分类的协方差矩阵相等，均为Σ，则任意两个分类的马氏距离之差为：

   W(x) = M_d(i) - M_d(j) = 2*(x - (μi+μj)/2).T * Σ.inv * (μi - μj)

   注意这个i, j的顺序会颠倒过来，所以这时候选取一个i作为分类结果时，它需要满足的条件是，对于所有j不等于i，
   W(x)均要大于零，与第一种情况恰恰相反

   因为此时判别函数W(x)变成了一个线性函数，所以称为线性判别分析，Linear Discriminant Analysis (LDA)

3. 还有一种判别分析称之为费歇判别 Fisher's Discriminant Analysis, 它的核心思想与一元方差分析类似，将K组p维数据
   投影到某一个方向，使得组与组之间的投影尽可能的分开，而每组之内的方差又要尽可能的小，换言之就是组间平方和要大
   组内平方和要小。

   假设方向向量为 a, 样本都是p维向量，则组间平方和和组内平方和分别用a.T * B * a 和 a.T * E * a 来表示，B与E都是
   n*n矩阵，具体算法不赘述

   那么就转化成一个最优化问题的求解
           a.T * B * a
   Δ(a) = -------------
           a.T * E * a

   可以转化成只含等式约束的最优化问题，即 constraint: a.T * E * a = 1, 求 Maximize(a.T * B * a)
   根据二阶充分必要条件，求导可得a为 E.inv * B的最大特征值对应的特征向量 
'''
