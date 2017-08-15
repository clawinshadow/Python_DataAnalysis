import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

x = np.linspace(0, 7, 200)

# scipy.stats.gamma里面的frozen只接受shape参数a，书中的rate参数b在这里等同于scale
rv1 = ss.gamma(1)
rv2 = ss.gamma(1.5)
rv3 = ss.gamma(2.0)

plt.figure()
plt.subplot()
plt.plot(x, rv1.pdf(x), 'b-', label='a=1.0, b=1.0')
plt.plot(x, rv2.pdf(x), 'r:', label='a=1.5, b=1.0')
plt.plot(x, rv3.pdf(x), 'k-.', label='a=2.0, b=1.0')
plt.title('Gamma Distributions')
plt.legend()

plt.show()
