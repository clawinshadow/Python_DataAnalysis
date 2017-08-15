import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

'''
beta分布，随机变量X的取值范围是[0, 1], 二项分布中参数p的共轭先验分布
因为a 和 b均在Gamma函数里面，所以它们的值必须大于零

                    gamma(a+b) * x**(a-1) * (1-x)**(b-1)
beta.pdf(x, a, b) = ------------------------------------
                             gamma(a)*gamma(b)

当a和b都小于1时，图形在a和b处呈现出两个尖峰，中间平滑
当a和b都等于1时，即均匀分布
当a和b都大于1时，单峰分布，a>b时右偏，反之则左偏
'''

x = np.linspace(0, 1, 200)

rv1 = ss.beta(0.1, 0.1)
rv2 = ss.beta(1, 1)
rv3 = ss.beta(2, 3)
rv4 = ss.beta(8, 4)

plt.figure()
plt.subplot()
plt.plot(x, rv1.pdf(x), 'b-', linewidth=2, label='a=0.1, b=0.1')
plt.plot(x, rv2.pdf(x), 'r:', linewidth=2, label='a=1, b=1')
plt.plot(x, rv3.pdf(x), 'k-.', linewidth=2, label='a=2, b=3')
plt.plot(x, rv4.pdf(x), 'g--', linewidth=2, label='a=8, b=4')
plt.legend(loc=2) # 将图例放在空白处
plt.ylim(0, 3)
plt.title('beta distribution')

plt.show()
