import scipy.linalg as sl
import numpy as np

print('''

  When SciPy is built using the optimized ATLAS LAPACK and BLAS libraries,
  it has very fast linear algebra capabilities.
  ---------------------------------------------------------------------------
  scipy.linalg contains all the functions in numpy.linalg.
  plus some other more advanced ones not contained in numpy.linalg
  -------------------------------------------------------------------------------
  numpy.matrix is matrix class that has a more convenient interface than numpy.ndarray
  for matrix operations.
  This class supports for example MATLAB-like creation syntax via the,
  has matrix multiplication as default for the * operator,
  and contains I and T members that serve as shortcuts for inverse and transpose
  --------------------------------------------------------------------------------
  Despite its convenience, the use of the numpy.matrix class is discouraged,
  since it adds nothing that cannot be accomplished with 2D numpy.ndarray objects,
  and may lead to a confusion of which class is being used

  ''')

print('Using np.matrix...')
A = np.mat('[1 2; 3 4]')
print('A: \n', A)
print('A.I: \n', A.I)

print('{0:-^60}'.format('Seperate Line'))

b = np.mat('[5 6]')
print('b: ', b)
print('b.T: \n', b.T)
print('A * b.T: \n', A * b.T)  # with shortcuts provided by np.matrix

print('{0:-^60}'.format('Seperate Line'))

print('Using np.ndarray...')
A = np.array([[1,2],[3,4]])
print('A: \n', A)
print('Inverse of A: \n', sl.inv(A))

print('{0:-^60}'.format('Seperate Line'))

b = np.array([[5,6]]) #2D array
print('b: ', b)
print('b.T: \n', b.T)
print('A * b: \n', A * b) # it's not inner product of vectors
print('A.dot(b.T): \n', A.dot(b.T)) # dot is the right function for matrix multiplication
print('{0:-^60}'.format('Seperate Line'))

b = np.array([5,6]) #1D array
print('b: ', b)
print('b.T: \n', b.T)  #not matrix transpose!
print('A.dot(b.T): \n', A.dot(b.T))  #does not matter for multiplication
