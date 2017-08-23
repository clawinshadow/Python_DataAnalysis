import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 200)
zeroOne = np.select([x <= 0, x > 0], [1, 0])
hinge = np.vectorize(lambda x: max(0, 1 - x))  # 两种不同的方式来简单的转化一个数组
logLoss = np.vectorize(lambda x: np.log2(1 + np.exp(-x)))

plt.figure()
plt.subplot()
plt.axis([-2.2, 2.2, -0.2, 3.3])
plt.xlabel(r'$\eta$')
plt.ylabel('loss')
plt.plot(x, zeroOne, 'k-', label='0-1')
plt.plot(x, hinge(x), color='midnightblue', linestyle=':', label='hinge')
plt.plot(x, logLoss(x), 'r--', label='logloss')

plt.legend()
plt.show()
