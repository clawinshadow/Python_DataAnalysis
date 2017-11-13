import numpy as np
import scipy.io as sio
import scipy.stats as ss
import matplotlib.pyplot as plt

# load data
data = sio.loadmat('kalmanTrackingDemo.mat')
print(data.keys())
x = data['x']
y = data['y']
print(x.shape, y.shape)
x = x.T
y = y.T
