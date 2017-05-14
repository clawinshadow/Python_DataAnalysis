关于MLE的详细情况，前面已经写过一篇文章，关于MAP的介绍，可以参见维基百科：https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation

本章着重介绍 MAP 和 MLE 的对比，MLE是频率主义学派即古典概率学派的参数估计方法，它的核心思想是模型参数是恒定不变的，只是等待我们通过样本去发现它；而MAP是贝叶斯学派的方法，它认为模型参数也是一个随机变量，有自己的概率分布，可以通过先根据经验给出一个模型参数的先验概率，然后再通过贝叶斯公司来计算最大化的后验概率，直觉上来说，举个例子，当我们刚接触某个女孩的时候，我们通过第一印象给出一个与她共度余生的概率，比如90%，这就是一个非常好的先验概率，然后通过多次约会与相处，这个过程就是收集观察数据，我们根据观察到的数据和先验概率，得出矫正后的后验概率，可能最后还有60%

简而言之，MLE：参数固定，数据集呈概率分布；MAP：数据集固定，参数是随机变量，呈某种概率分布，因为先验概率的引入，某种意义上来说MAP相当于MLE的正则化

我们假设 X 服从参数为 θ 的几何分布，即 X ~ Geometric(θ), 并且 θ 本身也是一个随机变量，服从以下概率分布(并不是严谨的概率分布，仅用于举例):

f(θ) = 2θ, if 0 < θ < 1; 0, otherwise. 这就是关于模型参数 θ 的先验概率。

那么我们做一次实验，收集到观测数据 X = 3，此时我们分别使用 MLE 和 MAP 来估计模型参数 θ：

MLE: 

因为 X 服从参数为 θ 的几何分布，那么似然函数 L(θ|X=3) = θ*(1-θ)**2, 从而 θ = argmax(L(θ|X=3))，求导可得 θ_MLE = 1/3

MAP:

后验概率 P(θ|X=3) = P(X=3|θ) * P(θ) / P(X=3), 我们先看P(X=3), 对于连续分布的随机变量θ来说，P(X=3) = integrate(P(X=3|θ)*P(θ)dθ), 这个是一个与参数θ无关的常量，所以在求argmax(P(θ|X=3))时，这个常量的分母可以略去，只需要球 argmax(P(X=3|θ) * P(θ))就可以了，而：

P(X=3|θ) = θ*(1-θ)**2, P(θ) = 2θ

所以原式等于 argmax(2 * θ**2 * (1-θ)**2), 求导可得最终结果 θ_MAP = 1/2