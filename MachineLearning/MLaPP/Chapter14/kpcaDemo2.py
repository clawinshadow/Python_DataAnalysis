import numpy as np
import scipy.io as sio
import scipy.linalg as sl
import sklearn.decomposition as sd
import matplotlib.pyplot as plt

# load data
data = sio.loadmat('kpcaDemo2.mat')
print(data.keys())
X = data['patterns']
print(X.shape)

