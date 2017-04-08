import datetime as dt
import numpy as np

'''
    This program compare the performance of doing vector addition by python way and numpy way
'''


def pythonsum(n):
    a = range(n)
    b = range(n)
    c = []
    for i in range(len(a)):
        c.append(a[i] ** 2 + b[i] ** 3)

    return c


def numpysum(n):
    a = np.arange(n, dtype=np.int64) ** 2
    b = np.arange(n, dtype=np.int64) ** 3
    c = a + b

    return c


def compare(n):
    start = dt.datetime.now()
    c = pythonsum(n)
    elapsed = dt.datetime.now() - start
    print('The last 2 elements of the sum: ', c[-2:])
    print('PythonSum ({0:d} elements) elapsed time in milliseconds: {1}'.format(n, elapsed.microseconds))

    start = dt.datetime.now()
    c = numpysum(n)
    elapsed = dt.datetime.now() - start
    print('The last 2 elements of the sum: ', c[-2:])
    print('NumpySum ({0:d} elements) elapsed time in milliseconds: {1}'.format(n, elapsed.microseconds))


compare(10000)
compare(20000)
compare(40000)

# results: much much faster in numpy
# The last 2 elements of the sum:  [999500079996, 999800010000]
# PythonSum (10000 elements) elapsed time in milliseconds: 15625
# The last 2 elements of the sum:  [999500079996 999800010000]
# NumpySum (10000 elements) elapsed time in milliseconds: 0
# The last 2 elements of the sum:  [7998000159996, 7999200020000]
# PythonSum (20000 elements) elapsed time in milliseconds: 31250
# The last 2 elements of the sum:  [7998000159996 7999200020000]
# NumpySum (20000 elements) elapsed time in milliseconds: 0
# The last 2 elements of the sum:  [63992000319996, 63996800040000]
# PythonSum (40000 elements) elapsed time in milliseconds: 53388
# The last 2 elements of the sum:  [63992000319996 63996800040000]
# NumpySum (40000 elements) elapsed time in milliseconds: 0
