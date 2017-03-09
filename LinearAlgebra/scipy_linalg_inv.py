import scipy.linalg as sl
import numpy as np


A = np.array([[1., 2.], [3., 4.]])
print('A: \n', A)
B = sl.inv(A)
print('Inverse of A: \n', B)
print('A * B: ', np.dot(A, B)) # should be I
