import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 200)
prior = ss.beta(20, 20).pdf(x) * 0.5 + ss.beta(10, 10).pdf(x) * 0.5
posterior = ss.beta(40, 30).pdf(x) * 0.346 + ss.beta(50, 20).pdf(x) * 0.654

plt.figure()
plt.subplot()
plt.title('mixture of Beta distributions')
plt.xlim(0, 1)
plt.ylim(0, 6)
plt.plot(x, prior, 'r--', label='prior')
plt.plot(x, posterior, color='darkblue', label='posterior')
plt.legend()

plt.show()
