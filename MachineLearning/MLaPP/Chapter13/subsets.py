import numpy as np

# 列出N的全部子集，总共2^N个
# 比如3的话就是 (), (0), (1), (2), (0, 1), (1, 2), (0, 2), (0, 1, 2)
def subsets(N):
    maxIter = 2**N
    result = dict()
    for i in range(maxIter):
        b = bin(i)
        s = str(b)[2:]
        s = s.rjust(N, '0') # ensure the length is N
        arr = np.array([(int)(x) for x in s])  # e.g. [1, 1, 0, 1, 0]
        count = np.count_nonzero(arr)          # 3
        indices = tuple(np.nonzero(arr)[0])    # (0, 1, 3)
        if count in result:
            result[count].add(indices)
        else:
            result[count] = set()
            result[count].add(indices)

    return result
