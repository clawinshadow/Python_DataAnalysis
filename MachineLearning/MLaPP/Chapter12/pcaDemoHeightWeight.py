import numpy as np
import scipy.io as sio
import sklearn.preprocessing as sp
import matplotlib.pyplot as plt
from pcaFit import *

# load data
data = sio.loadmat('heightWeight.mat')
X = data['heightWeightData'][:, 1:]
y = data['heightWeightData'][:, 0]
male = X[y == 1]
female = X[y == 2]

# fit without scale
mu, V, vr, z, x_recon = PCA(X)

fig = plt.figure(figsize=(13, 5))
fig.canvas.set_window_title('pcaDemoHeightWeight')

plt.subplot(121)
plt.xlabel('height')
plt.ylabel('weight')
plt.axis([55, 85, 50, 300])
plt.xticks(np.linspace(55, 85, 7))
plt.yticks(np.linspace(50, 300, 6))
plt.plot(male[:, 0], male[:, 1], 'bx', linestyle='none', ms=5)
plt.plot(female[:, 0], female[:, 1], 'ro', linestyle='none', fillstyle='none', ms=5)
plt.plot(x_recon[:, 0], x_recon[:, 1], 'k-')

# fit with scale
x = sp.StandardScaler().fit_transform(X)
male2 = x[y == 1]
female2 = x[y == 2]
mu2, V2, vr2, z2, x_recon2 = PCA(x)
print(vr2)

plt.subplot(122)
plt.xlabel('height')
plt.ylabel('weight')
plt.axis([-4, 4, -4, 5])
plt.xticks(np.linspace(-4, 4, 9))
plt.yticks(np.linspace(-4, 5, 10))
plt.plot(male2[:, 0], male2[:, 1], 'bx', linestyle='none', ms=5)
plt.plot(female2[:, 0], female2[:, 1], 'ro', linestyle='none', fillstyle='none', ms=5)
plt.plot(x_recon2[:, 0], x_recon2[:, 1], 'k-')
plt.plot([-5 * vr2.ravel()[0], 5 * vr2.ravel()[0]], [-5 * vr2.ravel()[1], 5 * vr2.ravel()[1]], 'k:')

plt.show()
